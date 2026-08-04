"""
Reduces frame-to-frame jitter in the mouth region of MuseTalk output.

MuseTalk regenerates only the mouth/jaw region per frame with no temporal
consistency constraint, so that region can flicker between plausible-but-
slightly-different shapes even when the rest of the frame (background, hair,
eyes) is pixel-stable. This detects exactly which pixels are actually moving
(via a prev/next-frame difference mask, not the current frame, so the mask
reflects genuine motion rather than the jitter it's meant to fix) and blends
toward the 3-frame temporal median only where doing so is likely to be a
correction rather than a smear.

A full-strength median blend looks fine for small jitter (the current frame
sits close to the local median), but on fast genuine mouth motion (e.g. quick
consonant transitions) the three frames can differ enough that per-pixel
median blending creates an anatomically-blended "superposition" of two mouth
shapes -- visible as ghosting/doubling at the teeth and lip edges. To avoid
that, the correction strength is scaled down the further the current frame
already is from the local median, so large genuine motion is left mostly
untouched while small jitter still gets damped.

Frames are written out as PNGs and encoded via the ffmpeg CLI rather than
cv2.VideoWriter, which has been observed to silently drop frames partway
through encoding on some inputs (cv2.VideoCapture reads all frames fine;
only the mp4v VideoWriter path loses them).

Per-frame blending is independent given precomputed neighbors, so it's
parallelized across CPU cores via a fork-based process pool: frames are read
into a module-global before the pool is created, so forked workers inherit
them via copy-on-write instead of re-serializing every frame array through
IPC (this box has 96 cores sitting idle during this stage otherwise).
"""
import glob
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

# Max fraction to blend toward the local median, even where jitter_confidence is 1.0.
BLEND_STRENGTH = 0.4
# Per-pixel median-vs-current deviation (0-255 scale) beyond which we treat the
# difference as genuine motion rather than jitter, and back off the correction.
JITTER_CAP = 25.0

_G = {}


def _motion_mask(prev_frame, next_frame, dilate_px=15, blur_px=21, threshold=12):
    diff = cv2.absdiff(prev_frame, next_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((dilate_px, dilate_px), np.uint8)
    mask = cv2.dilate(mask, kernel)
    mask = cv2.GaussianBlur(mask, (blur_px, blur_px), 0)
    return (mask.astype(np.float32) / 255.0)[..., None]


def _process_and_write(i):
    frames, work_dir = _G["frames"], _G["work_dir"]
    prev_f, cur_f, next_f = frames[i - 1], frames[i], frames[i + 1]
    mask = _motion_mask(prev_f, next_f)
    median = np.median(np.stack([prev_f, cur_f, next_f], axis=0), axis=0)
    cur_f32 = cur_f.astype(np.float32)

    median_diff = np.abs(median - cur_f32).mean(axis=2, keepdims=True)
    jitter_confidence = np.clip(1.0 - median_diff / JITTER_CAP, 0.0, 1.0)
    alpha = mask * BLEND_STRENGTH * jitter_confidence

    blended = median * alpha + cur_f32 * (1 - alpha)
    cv2.imwrite(os.path.join(work_dir, f"{i:08d}.png"), blended.astype(np.uint8))
    return i


def smooth_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 3:
        subprocess.run(["cp", input_path, output_path], check=True)
        return

    work_dir = tempfile.mkdtemp(prefix="temporal_smooth_")
    try:
        cv2.imwrite(os.path.join(work_dir, "00000000.png"), frames[0])
        cv2.imwrite(os.path.join(work_dir, f"{len(frames) - 1:08d}.png"), frames[-1])

        _G["frames"] = frames
        _G["work_dir"] = work_dir
        num_workers = min(32, multiprocessing.cpu_count())
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(executor.map(_process_and_write, range(1, len(frames) - 1), chunksize=4))

        written = len(glob.glob(os.path.join(work_dir, "*.png")))
        if written != len(frames):
            raise RuntimeError(
                f"temporal_smooth: expected {len(frames)} frames, only wrote {written}"
            )

        tmp_video = output_path + ".noaudio.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "warning", "-r", str(fps), "-f", "image2",
                "-i", os.path.join(work_dir, "%08d.png"),
                "-vcodec", "libx264", "-vf", "format=yuv420p", "-crf", "18",
                tmp_video,
            ],
            check=True, capture_output=True, text=True,
        )

        subprocess.run(
            [
                # NOTE: deliberately no -shortest here. MuseTalk's video output length
                # (frame_count / fps) is essentially always a few tens of ms shorter
                # than the driving TTS audio, since audio duration is rarely an exact
                # multiple of the frame period -- -shortest previously truncated the
                # AUDIO (not video) to match, silently clipping the last ~50-80ms of
                # actual spoken words off every render. Keeping full-length audio and
                # letting the last video frame hold for that final sliver instead is
                # far less noticeable than losing real speech content.
                "ffmpeg", "-y", "-v", "warning", "-i", tmp_video, "-i", input_path,
                "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy",
                "-c:a", "copy",
                output_path,
            ],
            check=True, capture_output=True, text=True,
        )
        os.remove(tmp_video)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _G.clear()


if __name__ == "__main__":
    smooth_video(sys.argv[1], sys.argv[2])
