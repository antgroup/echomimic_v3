"""
RunPod Serverless handler for the Muse avatar pipeline: F5-TTS voice clone ->
MuseTalk/Muse lip-sync -> GFPGAN+ESRGAN enhance/mouth-warp -> temporal smoothing.

Consolidates what used to be three separate warm HTTP servers (f5tts_server.py,
musetalk_server.py, enhance_server.py) into one process: a Serverless worker is
a single process anyway, so the HTTP-hop-between-services indirection that made
sense on a pod running all three concurrently buys nothing here. Model-loading
and inference logic below is ported near-verbatim from those three servers'
own _load() functions and route handlers; what changed is the transport layer
(FastAPI routes -> one handler() function) and path resolution (fixed
/workspace paths -> MUSE_DATA_DIR-relative, pointing at the attached Network
Volume instead of the pod's local disk).
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

# --- Path setup: MUST happen before any `musetalk.*` import. musetalk.utils.preprocessing
# initializes an mmpose model at IMPORT TIME using a path relative to cwd
# ('./models/dwpose/...'), and musetalk/utils/face_parsing loads its own weights the
# same way ('./models/face-parse-bisent/...') -- see those files for the exact lines.
# The original pod code satisfied this by keeping musetalk/ code and models/ weights
# as sibling directories and chdir-ing into their shared parent. Here code (musetalk/)
# ships in the image while weights (models/) live on the mounted Network Volume, so a
# symlink recreates the same "models/ next to musetalk/" layout the vendored code expects.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MUSE_DATA_DIR = os.environ.get("MUSE_DATA_DIR", "/runpod-volume")
os.chdir(APP_DIR)
sys.path.insert(0, APP_DIR)

_models_link = os.path.join(APP_DIR, "models")
if not os.path.islink(_models_link) and not os.path.exists(_models_link):
    os.symlink(os.path.join(MUSE_DATA_DIR, "models"), _models_link)

import copy

import cv2
import numpy as np
import runpod
import soundfile as sf
import torch
from transformers import WhisperModel

from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf

from f5_tts.infer.utils_infer import (
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)

from gfpgan import GFPGANer
from facexlib.parsing import init_parsing_model
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

from musetalk.utils.blending import get_image
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.utils import get_video_fps, datagen, load_all_model
from musetalk.utils.preprocessing import get_landmark_and_bbox, coord_placeholder

from temporal_smooth import smooth_video

# ---------------------------------------------------------------------------
# Config -- paths under MUSE_DATA_DIR (the attached RunPod Network Volume).
# See serverless/README.md for the exact directory layout expected here.
# ---------------------------------------------------------------------------
MODELS_DIR = os.path.join(MUSE_DATA_DIR, "models")
SERVICE_DIR = os.path.join(MUSE_DATA_DIR, "service")
AVATARS_CONFIG = os.path.join(SERVICE_DIR, "avatars.json")
VOICES_CONFIG = os.path.join(SERVICE_DIR, "voices.json")

MUSE_CHECKPOINT = os.path.join(MUSE_DATA_DIR, "checkpoints", "muse_v1", "unet-3000.pth")
UNET_CONFIG = os.path.join(MODELS_DIR, "musetalkV15", "musetalk.json")
WHISPER_DIR = os.path.join(MODELS_DIR, "whisper")
GFPGAN_MODEL = os.path.join(MODELS_DIR, "gfpgan", "GFPGANv1.4.pth")
ESRGAN_MODEL = os.path.join(MODELS_DIR, "realesrgan", "RealESRGAN_x4plus.pth")

# F5-TTS
F5_MODEL_NAME = "F5TTS_v1_Base"
F5_VOCODER_NAME = "vocos"

# MuseTalk / Muse
EXTRA_MARGIN = 10
AUDIO_PAD_LEFT = 2
AUDIO_PAD_RIGHT = 2
BATCH_SIZE = 8
PARSING_MODE = "jaw"
LEFT_CHEEK_WIDTH = 90
RIGHT_CHEEK_WIDTH = 90

# Enhance / mouth-amplitude warp (see enhance_server.py history for how these were tuned)
MOUTH_CLASSES = [11, 12, 13]
INTERIOR_CLASS = 11
FEATHER = 15
DILATE = 9
MOUTH_PAD = 25
PAD_RATIO = 0.35
MIN_PAD, MAX_PAD = 20, 70
OPEN_RATIO = 0.19
EXTRA_STRETCH = 1.32
EXTRA_STRETCH_X = 1.10
FALLOFF_EXP = 2.5
ABOVE_CENTER_BLEED = 0.06
MIN_INTERIOR_PIXELS = 25

PAUSE_MARKER_RE = re.compile(r"\[pause\]", re.IGNORECASE)
WORK_ROOT = tempfile.gettempdir()

_state = {}


# ---------------------------------------------------------------------------
# Cold-start model loading -- runs once per worker at import time, mirrors
# f5tts_server.py / musetalk_server.py / enhance_server.py's _load() functions.
# ---------------------------------------------------------------------------
def _load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading F5-TTS vocoder + model...")
    vocoder = load_vocoder(vocoder_name=F5_VOCODER_NAME, is_local=False, device="cuda")
    from importlib.resources import files
    model_cfg = OmegaConf.load(str(files("f5_tts").joinpath(f"configs/{F5_MODEL_NAME}.yaml")))
    model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
    model_arc = model_cfg.model.arch
    ckpt_file = str(cached_path(f"hf://SWivid/F5-TTS/{F5_MODEL_NAME}/model_1250000.safetensors"))
    f5_model = load_model(model_cls, model_arc, ckpt_file, mel_spec_type=F5_VOCODER_NAME, vocab_file="", device="cuda")

    print("Loading MuseTalk (Muse fine-tuned checkpoint)...")
    vae, unet, pe = load_all_model(
        unet_model_path=MUSE_CHECKPOINT, vae_type="sd-vae", unet_config=UNET_CONFIG, device=device
    )
    timesteps = torch.tensor([0], device=device)
    pe = pe.to(device)
    vae.vae = vae.vae.to(device)
    unet.model = unet.model.to(device)

    audio_processor = AudioProcessor(feature_extractor_path=WHISPER_DIR)
    weight_dtype = unet.model.dtype
    whisper = WhisperModel.from_pretrained(WHISPER_DIR).to(device=device, dtype=weight_dtype).eval()
    whisper.requires_grad_(False)

    fp = FaceParsing(left_cheek_width=LEFT_CHEEK_WIDTH, right_cheek_width=RIGHT_CHEEK_WIDTH)

    print("Loading GFPGAN + RealESRGAN + face-parsing...")
    restorer = GFPGANer(model_path=GFPGAN_MODEL, upscale=1, arch="clean", channel_multiplier=2, bg_upsampler=None)
    parse_net = init_parsing_model(model_name="bisenet", device=str(device))
    parse_net.eval()
    esrgan_arch = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    esrgan = RealESRGANer(
        scale=4, model_path=ESRGAN_MODEL, model=esrgan_arch,
        tile=0, tile_pad=10, pre_pad=0, half=(device.type == "cuda"), device=device,
    )

    _state.update(
        device=device,
        f5_vocoder=vocoder, f5_model=f5_model, f5_ref_cache={},
        vae=vae, unet=unet, pe=pe, timesteps=timesteps,
        audio_processor=audio_processor, whisper=whisper, weight_dtype=weight_dtype, fp=fp,
        restorer=restorer, parse_net=parse_net, esrgan=esrgan,
    )
    print("All models loaded. Worker ready.")


# ---------------------------------------------------------------------------
# Stage 1: TTS  (ported from f5tts_server.py)
# ---------------------------------------------------------------------------
def _synthesize(ref_audio_path, ref_text, gen_text, speed, output_path):
    cache_key = (ref_audio_path, ref_text)
    if cache_key not in _state["f5_ref_cache"]:
        _state["f5_ref_cache"][cache_key] = preprocess_ref_audio_text(ref_audio_path, ref_text)
    ref_audio, ref_text_processed = _state["f5_ref_cache"][cache_key]

    audio_segment, final_sample_rate, _ = infer_process(
        ref_audio, ref_text_processed, gen_text, _state["f5_model"], _state["f5_vocoder"],
        mel_spec_type=F5_VOCODER_NAME, speed=speed,
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sf.write(output_path, audio_segment, final_sample_rate)


def _concat_with_pauses(segment_paths, pause_seconds, output_path):
    if len(segment_paths) == 1:
        os.replace(segment_paths[0], output_path)
        return

    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=sample_rate,channels", "-of", "csv=p=0", segment_paths[0]],
        check=True, capture_output=True, text=True,
    )
    sample_rate, channels = probe.stdout.strip().split(",")
    silence_path = output_path + ".silence.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi",
         "-i", f"anullsrc=r={sample_rate}:cl={'mono' if channels == '1' else 'stereo'}",
         "-t", str(pause_seconds), silence_path],
        check=True, capture_output=True, text=True,
    )

    interleaved = []
    for i, seg in enumerate(segment_paths):
        interleaved.append(seg)
        if i < len(segment_paths) - 1:
            interleaved.append(silence_path)

    cmd = ["ffmpeg", "-y"]
    for f in interleaved:
        cmd += ["-i", f]
    filter_inputs = "".join(f"[{i}:a]" for i in range(len(interleaved)))
    cmd += ["-filter_complex", f"{filter_inputs}concat=n={len(interleaved)}:v=0:a=1[out]",
            "-map", "[out]", output_path]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    os.remove(silence_path)


# ---------------------------------------------------------------------------
# Stage 2: MuseTalk lip-sync  (ported from musetalk_server.py)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _lipsync(video_path, audio_path, output_path):
    device = _state["device"]
    vae, unet, pe, timesteps = _state["vae"], _state["unet"], _state["pe"], _state["timesteps"]
    audio_processor = _state["audio_processor"]
    whisper, weight_dtype, fp = _state["whisper"], _state["weight_dtype"], _state["fp"]

    work_dir = tempfile.mkdtemp(prefix="musetalk_req_", dir=WORK_ROOT)
    frames_dir = os.path.join(work_dir, "frames")
    result_frames_dir = os.path.join(work_dir, "result_frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(result_frames_dir, exist_ok=True)
    try:
        subprocess.run(
            ["ffmpeg", "-v", "fatal", "-i", video_path, "-start_number", "0",
             os.path.join(frames_dir, "%08d.png")],
            check=True,
        )
        input_img_list = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        fps = get_video_fps(video_path)

        whisper_input_features, librosa_length = audio_processor.get_audio_feature(audio_path)
        whisper_chunks = audio_processor.get_whisper_chunk(
            whisper_input_features, device, weight_dtype, whisper, librosa_length,
            fps=fps, audio_padding_length_left=AUDIO_PAD_LEFT, audio_padding_length_right=AUDIO_PAD_RIGHT,
        )

        coord_list, frame_list = get_landmark_and_bbox(input_img_list, 0)

        input_latent_list = []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == coord_placeholder:
                continue
            x1, y1, x2, y2 = bbox
            y2 = min(y2 + EXTRA_MARGIN, frame.shape[0])
            crop_frame = cv2.resize(frame[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
            input_latent_list.append(vae.get_latents_for_unet(crop_frame))

        frame_list_cycle = frame_list + frame_list[::-1]
        coord_list_cycle = coord_list + coord_list[::-1]
        input_latent_list_cycle = input_latent_list + input_latent_list[::-1]

        gen = datagen(
            whisper_chunks=whisper_chunks, vae_encode_latents=input_latent_list_cycle,
            batch_size=BATCH_SIZE, delay_frame=0, device=device,
        )
        res_frame_list = []
        for whisper_batch, latent_batch in gen:
            audio_feature_batch = pe(whisper_batch)
            latent_batch = latent_batch.to(dtype=unet.model.dtype)
            pred_latents = unet.model(latent_batch, timesteps, encoder_hidden_states=audio_feature_batch).sample
            res_frame_list.extend(vae.decode_latents(pred_latents))

        for i, res_frame in enumerate(res_frame_list):
            bbox = coord_list_cycle[i % len(coord_list_cycle)]
            ori_frame = copy.deepcopy(frame_list_cycle[i % len(frame_list_cycle)])
            x1, y1, x2, y2 = bbox
            y2 = min(y2 + EXTRA_MARGIN, ori_frame.shape[0])
            try:
                res_frame_resized = cv2.resize(
                    res_frame.astype(np.uint8), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4
                )
            except Exception:
                continue
            combine_frame = get_image(ori_frame, res_frame_resized, [x1, y1, x2, y2], mode=PARSING_MODE, fp=fp)
            cv2.imwrite(f"{result_frames_dir}/{i:08d}.png", combine_frame)

        temp_vid = os.path.join(work_dir, "temp.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "warning", "-r", str(fps), "-f", "image2",
             "-i", os.path.join(result_frames_dir, "%08d.png"),
             "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18", temp_vid],
            check=True,
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "warning", "-i", audio_path, "-i", temp_vid,
             "-c:v", "copy", "-c:a", "aac", output_path],
            check=True,
        )

        if not os.path.exists(output_path):
            raise RuntimeError("Lipsync produced no output video")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Stage 3: Fidelity enhancement + mouth-amplitude warp  (ported from enhance_server.py)
# ---------------------------------------------------------------------------
def _parse_face(img_bgr, parse_net, device):
    h, w = img_bgr.shape[:2]
    face_input = cv2.resize(img_bgr, (512, 512), interpolation=cv2.INTER_LINEAR)
    face_input = face_input[:, :, ::-1].astype(np.float32) / 255.0
    face_input = (face_input - 0.5) / 0.5
    face_input = torch.from_numpy(face_input.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = parse_net(face_input)[0]
    parsing = out.squeeze(0).argmax(0).cpu().numpy().astype(np.uint8)
    return cv2.resize(parsing, (w, h), interpolation=cv2.INTER_NEAREST)


def _warp_field(local_h, local_w, local_center_y, local_center_x, stretch_y, stretch_x):
    map_x, map_y = np.meshgrid(np.arange(local_w, dtype=np.float32), np.arange(local_h, dtype=np.float32))
    dy = map_y - local_center_y
    dx = map_x - local_center_x
    below = dy > 0

    dist_below = np.clip(dy / max(local_h - local_center_y, 1.0), 0, 1)
    dist_above = np.clip(-dy / max(local_center_y, 1.0), 0, 1)
    falloff_y = np.where(below, 1.0 - dist_below, 1.0 - dist_above)
    falloff_y = np.clip(falloff_y, 0.0, 1.0) ** FALLOFF_EXP

    dist_x = np.clip(np.abs(dx) / max(local_w / 2.0, 1.0), 0, 1)
    falloff_x = np.clip(1.0 - dist_x, 0.0, 1.0) ** FALLOFF_EXP

    stretch_field_y = np.where(
        below,
        1.0 + (stretch_y - 1.0) * falloff_y,
        1.0 + (stretch_y - 1.0) * falloff_y * ABOVE_CENTER_BLEED,
    )
    stretch_field_x = 1.0 + (stretch_x - 1.0) * falloff_y * falloff_x

    src_map_y = (local_center_y + dy / stretch_field_y).astype(np.float32)
    src_map_x = (local_center_x + dx / stretch_field_x).astype(np.float32)
    return src_map_x, src_map_y


def _warp_mouth_amplitude(img_bgr, combined_mask_raw, parsing_full):
    h, w = img_bgr.shape[:2]
    cys, cxs = np.where(combined_mask_raw > 127)
    if len(cxs) == 0:
        return img_bgr, combined_mask_raw
    mouth_bbox_w = cxs.max() - cxs.min()

    interior = (parsing_full == INTERIOR_CLASS).astype(np.uint8) * 255
    ys, xs = np.where(interior > 127)
    if len(ys) < MIN_INTERIOR_PIXELS:
        return img_bgr, combined_mask_raw

    y_top, y_bot = int(ys.min()), int(ys.max())
    x_left, x_right = int(xs.min()), int(xs.max())
    mouth_open_px = y_bot - y_top
    center_y = (y_top + y_bot) / 2.0
    center_x = (x_left + x_right) / 2.0

    reference_open_px = max(mouth_bbox_w * OPEN_RATIO, 8)
    warp_pad = int(np.clip(mouth_bbox_w * PAD_RATIO, MIN_PAD, MAX_PAD))

    openness_frac = np.clip(mouth_open_px / reference_open_px, 0.0, 1.0)
    stretch_y = 1.0 + (EXTRA_STRETCH - 1.0) * openness_frac
    stretch_x = 1.0 + (EXTRA_STRETCH_X - 1.0) * openness_frac

    wy0 = max(0, y_top - warp_pad)
    wy1 = min(h, y_bot + warp_pad)
    wx0 = max(0, x_left - warp_pad)
    wx1 = min(w, x_right + warp_pad)

    local_h, local_w = wy1 - wy0, wx1 - wx0
    local_center_y = center_y - wy0
    local_center_x = center_x - wx0
    src_map_x, src_map_y = _warp_field(local_h, local_w, local_center_y, local_center_x, stretch_y, stretch_x)

    warped_img = img_bgr.copy()
    warped_img[wy0:wy1, wx0:wx1] = cv2.remap(
        img_bgr[wy0:wy1, wx0:wx1], src_map_x, src_map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    warped_mask_raw = combined_mask_raw.copy()
    warped_mask_raw[wy0:wy1, wx0:wx1] = cv2.remap(
        combined_mask_raw[wy0:wy1, wx0:wx1], src_map_x, src_map_y,
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    return warped_img, warped_mask_raw


def _mouth_mask_and_bbox_from_raw(raw_mask, w, h):
    ys, xs = np.where(raw_mask > 127)
    if len(xs) == 0:
        bbox = None
    else:
        x1, x2 = max(0, xs.min() - MOUTH_PAD), min(w, xs.max() + MOUTH_PAD)
        y1, y2 = max(0, ys.min() - MOUTH_PAD), min(h, ys.max() + MOUTH_PAD)
        bbox = (x1, y1, x2, y2)

    if DILATE > 0:
        kernel = np.ones((DILATE, DILATE), np.uint8)
        raw_mask = cv2.dilate(raw_mask, kernel, iterations=1)
    feathered = cv2.GaussianBlur(raw_mask, (FEATHER, FEATHER), 0)
    return (feathered.astype(np.float32) / 255.0)[:, :, None], bbox


def _unsharp_mask(img, amount=2.2, sigma=1.3):
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = img.astype(np.float32) + amount * (img.astype(np.float32) - blurred.astype(np.float32))
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _sharpen_mouth(img_bgr, bbox, esrgan):
    if bbox is None:
        return img_bgr
    x1, y1, x2, y2 = bbox
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return img_bgr
    sharpened_crop, _ = esrgan.enhance(crop, outscale=1)
    sharpened_crop = _unsharp_mask(sharpened_crop)
    out = img_bgr.copy()
    out[y1:y2, x1:x2] = sharpened_crop
    return out


def _enhance(input_video, audio_path, output_path):
    device, restorer, parse_net, esrgan = _state["device"], _state["restorer"], _state["parse_net"], _state["esrgan"]

    work_dir = tempfile.mkdtemp(prefix="enhance_req_", dir=WORK_ROOT)
    frames_in = os.path.join(work_dir, "in")
    frames_out = os.path.join(work_dir, "out")
    os.makedirs(frames_in, exist_ok=True)
    os.makedirs(frames_out, exist_ok=True)
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", input_video],
            capture_output=True, text=True,
        )
        num, den = probe.stdout.strip().split("/")
        fps = float(num) / float(den)

        subprocess.run(
            ["ffmpeg", "-y", "-v", "warning", "-i", input_video, os.path.join(frames_in, "%08d.png")],
            check=True,
        )

        frame_files = sorted(glob.glob(os.path.join(frames_in, "*.png")))
        for f in frame_files:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            _, _, restored_img = restorer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
            if restored_img.shape[:2] != img.shape[:2]:
                restored_img = cv2.resize(restored_img, (img.shape[1], img.shape[0]))

            parsing_full = _parse_face(restored_img, parse_net, device)
            combined_mask_raw = np.isin(parsing_full, MOUTH_CLASSES).astype(np.uint8) * 255

            warped_img, warped_mask_raw = _warp_mouth_amplitude(restored_img, combined_mask_raw, parsing_full)

            mouth_mask, mouth_bbox = _mouth_mask_and_bbox_from_raw(warped_mask_raw, w, h)
            sharpened_mouth = _sharpen_mouth(warped_img, mouth_bbox, esrgan)

            composited = (
                sharpened_mouth.astype(np.float32) * mouth_mask
                + warped_img.astype(np.float32) * (1 - mouth_mask)
            )
            composited = np.clip(composited, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(frames_out, os.path.basename(f)), composited)

        temp_vid = os.path.join(work_dir, "temp.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "warning", "-r", str(fps), "-f", "image2",
             "-i", os.path.join(frames_out, "%08d.png"),
             "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18", temp_vid],
            check=True,
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-y", "-v", "warning", "-i", audio_path, "-i", temp_vid,
             "-c:v", "copy", "-c:a", "aac", output_path],
            check=True,
        )

        if not os.path.exists(output_path):
            raise RuntimeError("Enhancement produced no output video")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Orchestration + RunPod handler
# ---------------------------------------------------------------------------
def _load_avatars():
    with open(AVATARS_CONFIG) as f:
        return json.load(f)


def _load_voices():
    with open(VOICES_CONFIG) as f:
        return json.load(f)


def generate_video(avatar_name, script_text, voice_name=None, speed=1.0, pause_seconds=0.5, log=print):
    """Same stage sequence as pipeline.py's generate_video(), calling each
    stage's function directly in-process instead of over HTTP."""
    avatars = _load_avatars()
    if avatar_name not in avatars:
        raise ValueError(f"Unknown avatar: {avatar_name}. Available: {list(avatars.keys())}")
    cfg = avatars[avatar_name]

    if voice_name:
        voices = _load_voices()
        if voice_name not in voices:
            raise ValueError(f"Unknown voice: {voice_name}. Available: {list(voices.keys())}")
        voice_cfg = voices[voice_name]
    else:
        voice_name = avatar_name
        voice_cfg = {"voice_ref_path": cfg["voice_ref_path"], "voice_ref_text": cfg.get("voice_ref_text", "")}

    job_id = uuid.uuid4().hex[:12]
    job_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_", dir=WORK_ROOT)

    # --- Stage 1: TTS ---
    log(f"[{job_id}] Stage 1/4: TTS generation ({voice_name}'s voice)")
    t0 = time.time()
    tts_out_dir = os.path.join(job_dir, "tts")
    os.makedirs(tts_out_dir, exist_ok=True)

    segments = [s.strip() for s in PAUSE_MARKER_RE.split(script_text) if s.strip()]
    if not segments:
        segments = [script_text.strip()]

    segment_paths = []
    for i, segment_text in enumerate(segments):
        seg_path = os.path.join(tts_out_dir, f"segment_{i:03d}.wav")
        _synthesize(voice_cfg["voice_ref_path"], voice_cfg.get("voice_ref_text", ""), segment_text, speed, seg_path)
        if not os.path.exists(seg_path):
            raise RuntimeError(f"[{job_id}] TTS stage produced no output for segment {i}")
        segment_paths.append(seg_path)

    tts_audio_path = os.path.join(tts_out_dir, "tts_output.wav")
    _concat_with_pauses(segment_paths, pause_seconds, tts_audio_path)
    log(f"[{job_id}] Stage 1 done in {time.time()-t0:.1f}s ({len(segments)} segment(s), speed={speed}x)")

    # --- Stage 2: MuseTalk lip-sync ---
    log(f"[{job_id}] Stage 2/4: Lip-sync ({avatar_name})")
    t0 = time.time()
    lipsync_video_path = os.path.join(job_dir, "lipsync.mp4")
    _lipsync(cfg["video_path"], tts_audio_path, lipsync_video_path)
    log(f"[{job_id}] Stage 2 done in {time.time()-t0:.1f}s")

    # --- Stage 3: Fidelity enhancement ---
    log(f"[{job_id}] Stage 3/4: Fidelity enhancement")
    t0 = time.time()
    enhanced_video_path = os.path.join(job_dir, "enhanced.mp4")
    _enhance(lipsync_video_path, tts_audio_path, enhanced_video_path)
    log(f"[{job_id}] Stage 3 done in {time.time()-t0:.1f}s")

    # --- Stage 4: Temporal smoothing ---
    log(f"[{job_id}] Stage 4/4: Temporal smoothing")
    t0 = time.time()
    final_video_path = os.path.join(job_dir, "final.mp4")
    smooth_video(enhanced_video_path, final_video_path)
    log(f"[{job_id}] Stage 4 done in {time.time()-t0:.1f}s")

    return final_video_path, job_dir


def handler(event):
    """
    event["input"]: {avatar (required), script (required), voice, speed, pause_seconds}
    Returns {"video_base64": ...} on success, {"error": ...} on failure. Videos in
    this pipeline run ~10-20MB, well within a synchronous base64 response -- no
    separate storage/retrieval step needed at this scale (see plan doc if that changes).
    """
    import base64

    inp = event.get("input", {})
    avatar_name = inp.get("avatar")
    script_text = inp.get("script")
    if not avatar_name or not script_text:
        return {"error": "Both 'avatar' and 'script' are required."}

    job_dir = None
    try:
        final_video_path, job_dir = generate_video(
            avatar_name, script_text,
            voice_name=inp.get("voice"),
            speed=float(inp.get("speed", 1.0)),
            pause_seconds=float(inp.get("pause_seconds", 0.5)),
        )
        with open(final_video_path, "rb") as f:
            video_b64 = base64.b64encode(f.read()).decode("ascii")
        return {"video_base64": video_b64, "avatar": avatar_name, "voice": inp.get("voice") or avatar_name}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if job_dir:
            shutil.rmtree(job_dir, ignore_errors=True)


if __name__ == "__main__":
    # Model loading is gated here (not run unconditionally at module level) so that
    # merely importing this file -- e.g. a static handler-detection check trying to
    # introspect the module without a GPU or model volume available -- doesn't trigger
    # a full model load and crash. Mirrors the deferred-startup pattern the original
    # three servers used (@app.on_event("startup")), which only fired once uvicorn
    # actually started serving, not at import time.
    _load_models()
    runpod.serverless.start({"handler": handler})
