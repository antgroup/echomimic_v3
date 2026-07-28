# RunPod Serverless worker for the Muse avatar pipeline (F5-TTS voice clone ->
# MuseTalk/Muse lip-sync -> GFPGAN+ESRGAN enhance/mouth-warp -> temporal smoothing).
#
# Base image matches the pod this was developed on exactly, so torch/CUDA already
# present there (torch==2.4.1+cu124) is reused as-is rather than reinstalled --
# see requirements.txt for why torch/torchvision/torchaudio/triton are omitted there.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# NOTE: RunPod's GitHub-deploy build always uses the repo ROOT as the build
# context, even though "Dockerfile Path" points at serverless/Dockerfile --
# so every COPY source below is repo-root-relative (serverless/...), not
# relative to this Dockerfile's own directory. Without the serverless/
# prefix, these either fail outright or (worse, for requirements.txt) silently
# match an unrelated same-named file at the repo root instead.
COPY serverless/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Vendored MuseTalk package source (code only -- model weights live on the
# attached Network Volume, not in the image; see README.md).
COPY serverless/musetalk/ ./musetalk/

COPY serverless/handler.py serverless/temporal_smooth.py ./

# Where the Network Volume is mounted by RunPod Serverless workers.
ENV MUSE_DATA_DIR=/runpod-volume

CMD ["python3", "-u", "handler.py"]
