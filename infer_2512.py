import os
import sys
import argparse

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer


from src.dist import set_multi_gpus_devices, shard_model
from src.wan_vae import AutoencoderKLWan
from src.wan_image_encoder import  CLIPModel
from src.wan_text_encoder import  WanT5EncoderModel
from src.wan_transformer3d_audio_2512 import WanTransformerAudioMask3DModel
from src.pipeline_wan_fun_inpaint_audio_2512 import WanFunInpaintAudioPipeline

from src.utils import (filter_kwargs, get_image_to_video_latent, get_image_to_video_latent2,
                                   save_videos_grid)

from src.fm_solvers import FlowDPMSolverMultistepScheduler
from src.fm_solvers_unipc import FlowUniPCMultistepScheduler
from src.cache_utils import get_teacache_coefficients

import decord
import json
import random
import math

import librosa
from moviepy import VideoFileClip, AudioFileClip
import pyloudnorm as pyln
from transformers import Wav2Vec2FeatureExtractor
from videox_fun.models.wav2vec2 import Wav2Vec2Model
from einops import rearrange

def parse_args():
    parser = argparse.ArgumentParser(description="WanFun Inference")
    
    # Model paths and config
    parser.add_argument("--config_path", type=str, default="config/wan2.1/wan_civitai.yaml", help="Config path")
    parser.add_argument("--model_name", type=str, default="Wan2.1-Fun-V1.1-1.3B-InP", help="Model name")
    parser.add_argument("--ckpt_idx", type=int, default=50000, help="Checkpoint index")
    parser.add_argument("--transformer_path", type=str, default="", help="Transformer path")
    parser.add_argument("--vae_path", type=str, default=None, help="VAE path")
    parser.add_argument("--lora_path", type=str, default=None, help="LoRA path")
    parser.add_argument("--save_path", type=str, default="outputs", help="Save path")
    
    # Audio model path
    parser.add_argument("--wav2vec_model_dir", type=str, default="chinese-wav2vec2-base", help="Wav2Vec model directory")
    
    # Input paths
    parser.add_argument("--image_path", type=str, required=True, help="Input image path")
    parser.add_argument("--audio_path", type=str, required=True, help="Input audio path")
    
    # Inference parameters
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--sampler_name", type=str, default="Flow_Unipc", choices=["Flow", "Flow_Unipc", "Flow_DPM++"], help="Sampler name")
    parser.add_argument("--video_length", type=int, default=81, help="Video length")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale")
    parser.add_argument("--audio_guidance_scale", type=float, default=3.0, help="Audio guidance scale")
    parser.add_argument("--audio_scale", type=float, default=1.0, help="Audio scale")
    parser.add_argument("--neg_scale", type=float, default=1.0, help="Negative scale")
    parser.add_argument("--neg_steps", type=int, default=0, help="Negative steps")
    parser.add_argument("--num_inference_steps", type=int, default=25, help="Number of inference steps")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--lora_weight", type=float, default=0.6, help="LoRA weight")
    
    # TeaCache parameters
    parser.add_argument("--enable_teacache", action="store_true", default=True, help="Enable TeaCache")
    parser.add_argument("--teacache_threshold", type=float, default=0.1, help="TeaCache threshold")
    parser.add_argument("--num_skip_start_steps", type=int, default=5, help="Number of skip start steps")
    parser.add_argument("--teacache_offload", action="store_true", default=False, help="TeaCache offload")
    
    # Dynamic CFG
    parser.add_argument("--use_dynamic_cfg", action="store_true", default=False, help="Use dynamic CFG")
    parser.add_argument("--use_dynamic_acfg", action="store_true", default=False, help="Use dynamic audio CFG")
    
    # Riflex
    parser.add_argument("--enable_riflex", action="store_true", default=False, help="Enable Riflex")
    parser.add_argument("--riflex_k", type=int, default=6, help="Riflex k")
    
    # Mask
    parser.add_argument("--use_un_ip_mask", action="store_true", default=False, help="Use un IP mask")
    
    # GPU and memory
    parser.add_argument("--GPU_memory_mode", type=str, default="sequential_cpu_offload", help="GPU memory mode")
    parser.add_argument("--ulysses_degree", type=int, default=1, help="Ulysses degree")
    parser.add_argument("--ring_degree", type=int, default=1, help="Ring degree")
    parser.add_argument("--fsdp_dit", action="store_true", default=False, help="FSDP DIT")
    parser.add_argument("--weight_dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"], help="Weight dtype")
    
    # Other parameters
    parser.add_argument("--sample_size", type=int, nargs=2, default=[768, 768], help="Sample size")
    parser.add_argument("--fps", type=int, default=25, help="FPS")
    parser.add_argument("--add_prompt", type=str, default="", help="Additional prompt")
    parser.add_argument("--negative_prompt", type=str, default="Gesture is bad. Gesture is unclear. Strange and twisted hands. Bad hands. Bad fingers. Unclear and blurry hands. Unclear gestures, broken hands, fused fingers. 手指融合，", help="Negative prompt")
    parser.add_argument("--mouth_prompts", type=str, default=None, help="Mouth prompts")
    
    # Skip ratio
    parser.add_argument("--cfg_skip_ratio", type=float, default=0.0, help="CFG skip ratio")
    parser.add_argument("--shift", type=float, default=5.0, help="Shift value")
    
    return parser.parse_args()

def get_sample_size(pil_img, sample_size):
    w, h = pil_img.size
    ori_a = w * h
    default_a = sample_size[0] * sample_size[1]
    if default_a < ori_a:
        ratio_a = math.sqrt(ori_a / sample_size[0] / sample_size[1])

        w = w / ratio_a // 16 * 16
        h = h / ratio_a // 16 * 16
    else:
        w = w // 16 * 16
        h = h // 16 * 16

    return [int(h), int(w)]


def get_ip_mask(coords):
    y1, y2, x1, x2, h, w = coords
    Y, X = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    mask = (Y.unsqueeze(-1) >= y1) & (Y.unsqueeze(-1) < y2) & (X.unsqueeze(-1) >= x1) & (X.unsqueeze(-1) < x2)
    
    mask = mask.reshape(-1)
    return mask.float()

def get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length, sr=16000, fps=25, device='cpu'):

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(wav2vec_feature_extractor(mel_input, sampling_rate=sr).input_values)
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def main():
    args = parse_args()
    
    # Assign args to original variables
    config_path = args.config_path
    model_name = args.model_name
    ckpt_idx = args.ckpt_idx
    transformer_path = args.transformer_path
    vae_path = args.vae_path
    lora_path = args.lora_path
    save_path = args.save_path
    wav2vec_model_dir = args.wav2vec_model_dir
    image_path = args.image_path
    audio_path = args.audio_path
    prompt = args.prompt
    sampler_name = args.sampler_name
    video_length = args.video_length
    guidance_scale = args.guidance_scale
    audio_guidance_scale = args.audio_guidance_scale
    audio_scale = args.audio_scale
    neg_scale = args.neg_scale
    neg_steps = args.neg_steps
    num_inference_steps = args.num_inference_steps
    seed = args.seed
    lora_weight = args.lora_weight
    enable_teacache = args.enable_teacache
    teacache_threshold = args.teacache_threshold
    num_skip_start_steps = args.num_skip_start_steps
    teacache_offload = args.teacache_offload
    use_dynamic_cfg = args.use_dynamic_cfg
    use_dynamic_acfg = args.use_dynamic_acfg
    enable_riflex = args.enable_riflex
    riflex_k = args.riflex_k
    use_un_ip_mask = args.use_un_ip_mask
    GPU_memory_mode = args.GPU_memory_mode
    ulysses_degree = args.ulysses_degree
    ring_degree = args.ring_degree
    fsdp_dit = args.fsdp_dit
    weight_dtype_str = args.weight_dtype
    sample_size = args.sample_size
    fps = args.fps
    add_prompt = args.add_prompt
    negative_prompt = args.negative_prompt
    mouth_prompts = args.mouth_prompts
    cfg_skip_ratio = args.cfg_skip_ratio
    shift = args.shift
    
    # Convert weight dtype
    weight_dtype = torch.bfloat16 if weight_dtype_str == "bfloat16" else torch.float16

    # Load audio models
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_model_dir, local_files_only=True).to('cpu')
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_model_dir, local_files_only=True)

    device = set_multi_gpus_devices(ulysses_degree, ring_degree)
    config = OmegaConf.load(config_path)

    transformer = WanTransformer.from_pretrained(
        os.path.join(model_name, config['transformer_additional_kwargs'].get('transformer_subpath', 'transformer')),
        transformer_additional_kwargs=OmegaConf.to_container(config['transformer_additional_kwargs']),
        low_cpu_mem_usage=True if not fsdp_dit else False,
        torch_dtype=weight_dtype,
    )

    if transformer_path is not None:
        
        print(f"From checkpoint: {transformer_path}")
        if transformer_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(transformer_path)
        else:
            state_dict = torch.load(os.path.join(transformer_path, f'checkpoint-{ckpt_idx}.pth'))
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = transformer.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Vae
    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_name, config['vae_kwargs'].get('vae_subpath', 'vae')),
        additional_kwargs=OmegaConf.to_container(config['vae_kwargs']),
    ).to(weight_dtype)

    if vae_path is not None:
        print(f"From checkpoint: {vae_path}")
        if vae_path.endswith("safetensors"):
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(vae_path)
        else:
            state_dict = torch.load(vae_path, map_location="cpu")
        state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

        m, u = vae.load_state_dict(state_dict, strict=False)
        print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")

    # Get Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('tokenizer_subpath', 'tokenizer')),
    )

    # Get Text encoder
    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_name, config['text_encoder_kwargs'].get('text_encoder_subpath', 'text_encoder')),
        additional_kwargs=OmegaConf.to_container(config['text_encoder_kwargs']),
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    )
    text_encoder = text_encoder.eval()

    # Get Clip Image Encoder
    clip_image_encoder = CLIPModel.from_pretrained(
        os.path.join(model_name, config['image_encoder_kwargs'].get('image_encoder_subpath', 'image_encoder')),
    ).to(weight_dtype)
    clip_image_encoder = clip_image_encoder.eval()

    # Get Scheduler
    Choosen_Scheduler = scheduler_dict = {
        "Flow": FlowMatchEulerDiscreteScheduler,
        "Flow_Unipc": FlowUniPCMultistepScheduler,
        "Flow_DPM++": FlowDPMSolverMultistepScheduler,
    }[sampler_name]
    if sampler_name == "Flow_Unipc" or sampler_name == "Flow_DPM++":
        config['scheduler_kwargs']['shift'] = 1
    scheduler = Choosen_Scheduler(
        **filter_kwargs(Choosen_Scheduler, OmegaConf.to_container(config['scheduler_kwargs']))
    )

    # Get Pipeline
    pipeline = WanFunInpaintAudioPipeline(
        transformer=transformer,
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder
    )

    if ulysses_degree > 1 or ring_degree > 1:
        from functools import partial
        transformer.enable_multi_gpus_inference()
        if fsdp_dit:
            shard_fn = partial(shard_model, device_id=device, param_dtype=weight_dtype)
            pipeline.transformer = shard_fn(pipeline.transformer)


    pipeline.to(device=device)

    coefficients = get_teacache_coefficients(model_name) if enable_teacache else None
    if coefficients is not None:
        print(f"Enable TeaCache with threshold {teacache_threshold} and skip the first {num_skip_start_steps} steps.")
        pipeline.transformer.enable_teacache(
            coefficients, num_inference_steps, teacache_threshold, num_skip_start_steps=num_skip_start_steps, offload=teacache_offload
        )

    generator = torch.Generator(device=device).manual_seed(seed)

    if lora_path is not None:
        pipeline = merge_lora(pipeline, lora_path, lora_weight)

    pipeline.to(device=device)

    # Create output directory
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    with torch.no_grad():
        # Process single image and audio
        print(f"Processing: {image_path}")
        print(f"Audio: {audio_path}")
        print(f"Prompt: {prompt}")
        
        # Generate output filename
        image_name = os.path.basename(image_path).split('.')[0]
        output_video_path = os.path.join(save_path, f"{image_name}_output.mp4")
        
        # Check if output already exists
        if os.path.exists(output_video_path):
            print(f"⏭️  Skip: {output_video_path} already exists.")
            return

        # Load reference image
        ref_image = Image.open(image_path).convert("RGB")
        ref_start = np.array(ref_image)

        # Load audio
        audio_clip = AudioFileClip(audio_path)
        video_length_actual = min(int(audio_clip.duration * fps), video_length)
        video_length_actual = int((video_length_actual - 1) // vae.config.temporal_compression_ratio * vae.config.temporal_compression_ratio) + 1 if video_length_actual != 1 else 1

        # Get audio features
        mel_input, sr = librosa.load(audio, sr=16000)
        mel_input = loudness_norm(mel_input, sr)
        mel_input = mel_input[:int(video_length_actual / 25 * sr)]
        
        print(f"Audio length: {int(len(mel_input)/ sr * 25)}, Video length: {video_length_actual}")
        audio_feature_wav2vec = get_audio_embed(mel_input, wav2vec_feature_extractor, audio_encoder, video_length_actual, sr=16000, fps=25, device='cpu')
        
        # Get audio batch 
        audio_embeds = audio_feature_wav2vec.to(device=device, dtype=weight_dtype)
        
        indices = (torch.arange(2 * 2 + 1) - 2) * 1 
        center_indices = torch.arange(
            0,  
            video_length_actual,
            1,).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=audio_embeds.shape[0]-1)
        audio_embeds = audio_embeds[center_indices] # F w s c [F, 5, 12, 768]
        audio_embeds = audio_embeds.unsqueeze(0).to(device=device)

        print(f"Audio embeds shape: {audio_embeds.shape}")

        validation_image_start = Image.fromarray(ref_start).convert("RGB")
        validation_image_end = None
        latent_frames = (video_length_actual - 1) // vae.config.temporal_compression_ratio + 1

        if enable_riflex:
            pipeline.transformer.enable_riflex(k = riflex_k, L_test = latent_frames)
        sample_size_0, sample_size_1 = get_sample_size(validation_image_start, sample_size)

        input_video, input_video_mask, clip_image = get_image_to_video_latent2(validation_image_start, validation_image_end, video_length=video_length_actual, sample_size=[sample_size_0, sample_size_1])

        sample = pipeline(
            prompt, 
            num_frames = video_length_actual,
            negative_prompt = negative_prompt,
            audio_embeds = audio_embeds,
            audio_scale=audio_scale,
            ip_mask = None,
            use_un_ip_mask=use_un_ip_mask,
            height      = sample_size_0,
            width       = sample_size_1,
            generator   = generator,
            neg_scale = neg_scale,
            neg_steps = neg_steps,
            use_dynamic_cfg=use_dynamic_cfg,
            use_dynamic_acfg=use_dynamic_acfg,
            guidance_scale = guidance_scale,
            audio_guidance_scale = audio_guidance_scale,
            num_inference_steps = num_inference_steps,
            video      = input_video,
            mask_video   = input_video_mask,
            clip_image = clip_image,
            cfg_skip_ratio = cfg_skip_ratio,
            shift = shift,
        ).videos

        # Save temporary video
        tmp_video_path = os.path.join(save_path, f"{image_name}_tmp.mp4")
        save_videos_grid(sample[:,:,:video_length_actual], tmp_video_path, fps=fps)
        
        # Add audio to video
        video_clip = VideoFileClip(tmp_video_path)
        audio_clip = audio_clip.subclipped(0, video_length_actual / fps)
        video_clip = video_clip.with_audio(audio_clip)
        video_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", threads=2)

        # Clean up temporary file
        os.remove(tmp_video_path)
        print(f"✅ Saved output to: {output_video_path}")

if __name__ == "__main__":
    main()
