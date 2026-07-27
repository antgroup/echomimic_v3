# Muse avatar pipeline — RunPod Serverless deployment

This directory is unrelated to the rest of this repo (the upstream EchoMimicV3 /
Wan-diffusion demo). It deploys Skill Studio AI's own avatar pipeline — F5-TTS
voice cloning -> MuseTalk/Muse lip-sync -> GFPGAN+ESRGAN enhancement and
mouth-amplitude warp -> temporal smoothing — as a RunPod Serverless endpoint
that scales to zero workers when idle, instead of an always-on pod.

The `musetalk/` package here is vendored from [Tencent's MuseTalk](https://github.com/TMElyralab/MuseTalk)
(MIT license, see `musetalk/LICENSE`) — code only, no model weights.

## Why Serverless instead of a pod

With most traffic being free-tier/low-volume, an always-on pod bills for idle
hours regardless of whether anyone is generating a video. A Serverless worker
with `workersMin: 0` costs nothing while idle and spins up only on request,
billed per second of actual compute. Measured cost per video on an A40:
~6.8 min of compute at ~$0.44/hr, i.e. roughly $0.05/video — the real unit
cost, versus renting a GPU 24/7 whether or not it's used.

## Architecture

`handler.py` consolidates three warm HTTP servers that used to run on the pod
(`f5tts_server.py`, `musetalk_server.py`, `enhance_server.py`, orchestrated by
`pipeline.py`) into one process. A Serverless worker is a single process
anyway, so the HTTP-hop-between-services indirection that made sense when all
three ran concurrently on one pod buys nothing here — model loading and
inference logic is ported over near-verbatim, just called directly in-process
instead of over `localhost` HTTP.

**Code vs. data split**: the Docker image contains code only (this directory's
`handler.py`, `temporal_smooth.py`, the vendored `musetalk/` package, and pip
dependencies). Model checkpoints, avatar reference videos, and voice clips
live on an attached RunPod **Network Volume** (mounted at `/runpod-volume` by
default, configurable via the `MUSE_DATA_DIR` env var) — never in this
(public) git repo.

## Network Volume layout

The volume must be laid out as:

```
/runpod-volume/
  models/
    musetalkV15/{musetalk.json, unet.pth}     # stock MuseTalk v1.5 config + weights
    sd-vae/                                    # VAE weights
    whisper/                                   # Whisper feature extractor
    dwpose/dw-ll_ucoco_384.pth                  # pose/landmark detector
    face-parse-bisent/{resnet18-*.pth, 79999_iter.pth}
    gfpgan/GFPGANv1.4.pth
    realesrgan/RealESRGAN_x4plus.pth
  checkpoints/
    muse_v1/unet-3000.pth                       # the fine-tuned Muse checkpoint
  service/
    avatars.json
    voices.json
    videos/                                     # per-avatar reference videos
    voices/                                      # per-avatar reference voice clips
```

At startup, `handler.py` symlinks `./models` (relative to the code directory)
to `{MUSE_DATA_DIR}/models` before importing anything from `musetalk.*` —
`musetalk/utils/preprocessing.py` and `musetalk/utils/face_parsing` both load
their own weights from a `./models/...` path relative to the process's cwd at
**import time**, a holdover from the original pod layout where `musetalk/`
code and `models/` weights sat as sibling directories. The symlink recreates
that sibling relationship without duplicating model weights into the image.

`facexlib`'s own bisenet parsing weight (used by the enhance stage, distinct
from MuseTalk's own face-parsing model above) auto-downloads on first use if
not already cached — no separate migration step needed for it. Likewise the
F5-TTS base model (`model_1250000.safetensors`) downloads from Hugging Face at
cold start via `cached_path()` — it wasn't cached on the pod either, so this
isn't a new regression, just worth knowing it adds to cold-start latency.

## Request / response

```jsonc
// input
{
  "avatar": "Lyselle",        // required, must exist in avatars.json
  "script": "Hello there!",   // required; "[pause]" markers insert silence
  "voice": null,               // optional, defaults to avatar's own voice
  "speed": 1.0,                 // optional, F5-TTS speech-rate multiplier
  "pause_seconds": 0.5          // optional, silence duration per [pause] marker
}
```

```jsonc
// output
{ "video_base64": "...", "avatar": "Lyselle", "voice": "Lyselle" }
// or on failure:
{ "error": "..." }
```

Videos in this pipeline run ~10-20MB, comfortably within a synchronous
base64 response — no separate object-storage upload/retrieval step at this
scale.

## Deploying

1. Connect this GitHub repo to RunPod (RunPod dashboard -> Serverless -> New
   Endpoint -> Deploy from GitHub Repo -> point at this `serverless/`
   directory's Dockerfile). This is a one-time GitHub App authorization done
   in RunPod's dashboard, not available via API.
2. Create and populate the Network Volume per the layout above, in the same
   data center you select for the endpoint (cross-region volume attachment
   isn't supported).
3. Configure the endpoint: GPU pool matching A40 (`AMPERE_48`), attach the
   Network Volume, `workersMin: 0` (scale-to-zero), a small `workersMax` (2-3)
   for burst headroom, and a short idle timeout so a quick burst of requests
   reuses a still-warm worker instead of cold-starting every time.
4. Every push to the connected branch rebuilds and redeploys automatically.

Out of scope for this deployment (left as a later, application-layer
decision): free-tier vs. paid-tier request routing, a reserved always-warm
worker for paying users, and any request queueing beyond RunPod's own
autoscaler.
