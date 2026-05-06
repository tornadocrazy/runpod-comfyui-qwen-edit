FROM runpod/worker-comfyui:5.8.5-base
# build trigger: 2026-05-07T02:00 — SageAttention + HF_HOME + drop Manager + lighter GFPGAN

# uv is pre-installed in the base image; point it at the base venv
ENV VIRTUAL_ENV=/opt/venv

# CUDA arches our endpoint allows: Ada (L40 sm_89), Hopper (H200 sm_90),
# Blackwell (RTX PRO 6000 sm_120). Tells torch + sageattention which kernels
# to JIT/compile for so we don't fall back to slower paths at runtime.
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"

# Skipped apt-get update + install — base image already has wget; unzip is
# replaced with a Python stdlib call below for the one .zip we need to extract.
# Drops ~30s from every build.

# Install hf_transfer (Rust-based parallel multi-connection downloader) and
# enable it for all huggingface-cli downloads. Big files (14 GB diffusion,
# 7 GB text encoder) drop from 1-5 min on wget single-connection to ~30-60s
# consistently with 8-16 parallel chunks.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install hf_transfer
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Point all HF-aware tools (transformers, diffusers, insightface auto-download,
# any future LoRA URL fetches) at RunPod's Model Cache mount point. Means a
# single shared cache across all tools instead of per-tool re-downloads.
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV HF_HUB_CACHE=/runpod-volume/huggingface-cache
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache

# SageAttention — int8 attention kernels, 1.3-2× faster than PyTorch SDPA on
# H100/A100. ComfyUI picks it up via --use-sage-attention flag in start.sh.
# Note: prebuilt wheel for torch 2.10 may not exist — install with || true so
# image build succeeds either way; ComfyUI falls back to default attention.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install sageattention || echo "sageattention install failed — fallback to default attention"

# ─────────────────────────────────────────────────────────────────────────────
# Python dependencies for ReActor face swap
# Pre-built insightface wheel avoids C compilation issues.
# Split into two installs: the wheel uses --no-deps (its transitive deps —
# numpy, scipy, cython, etc. — are all in the base image already), so uv
# doesn't pull them into the resolver graph. Then a smaller separate install
# for the pinned packages that need their dep trees resolved.
# ─────────────────────────────────────────────────────────────────────────────
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps \
    https://huggingface.co/iwr-redmond/linux-wheels/resolve/main/insightface-0.7.3-cp312-cp312-linux_x86_64.whl

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
    onnxruntime-gpu==1.22.0 \
    facexlib==0.3.0 \
    gfpgan==1.3.8 \
    basicsr==1.4.2 \
    filterpy==1.4.5 \
    tb-nightly \
    numba \
    llvmlite

# ─────────────────────────────────────────────────────────────────────────────
# Custom nodes required by Qwen Edit workflows
# ─────────────────────────────────────────────────────────────────────────────
# Clone all nodes in one step (skips comfy-node-install registry overhead)
# Locked workflow uses: KJNodes (CFGNorm) + ReActor (face swap) only.
# Dropped: comfyui_controlnet_aux (we send pre-rendered DWPose images, no preprocessor needed),
#          comfyui_qwen_image_edit_adv (reverted from Adv encoder, locked on built-in TextEncodeQwenImageEditPlus),
#          ComfyUI-Inpaint-CropAndStitch (using green-screen inpaint instead).
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes && \
    git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor comfyui-reactor

# One uv resolver pass for all node requirements + pin GPU onnxruntime
# After node deps drag in CPU onnxruntime, force a clean reinstall of -gpu so
# insightface gets a working InferenceSession (otherwise CPU/GPU share the
# same `onnxruntime/` package dir and the uninstall corrupts the namespace).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt \
        -r /comfyui/custom_nodes/comfyui-reactor/requirements.txt && \
    uv pip uninstall onnxruntime onnxruntime-gpu && \
    uv pip install --reinstall onnxruntime-gpu==1.22.0 && \
    python -c "import onnxruntime; assert hasattr(onnxruntime, 'InferenceSession'), 'onnxruntime broken'; print('onnxruntime OK:', onnxruntime.__version__)"

# Bypass ReActor NSFW filter — avoids downloading large classifier at runtime
COPY reactor_sfw.py /comfyui/custom_nodes/comfyui-reactor/scripts/reactor_sfw.py

# ReActor source-face pre-analysis plugin — kicks off InsightFace analysis in
# a background thread when a prompt arrives, populating ReActor's MD5 cache
# before KSampler finishes. Saves ~4s per job (parallel with KSampler).
COPY reactor_preanalyze /comfyui/custom_nodes/reactor_preanalyze

# Remove unused nodes to speed up startup
RUN rm -rf /comfyui/comfy_api_nodes

# Drop ComfyUI-Manager — base image bakes it in but we don't use the UI for
# anything (workflow comes via API). Manager runs a security/registry scan
# at boot adding ~3s of cold-start time we don't need.
RUN rm -rf /comfyui/custom_nodes/ComfyUI-Manager

# ─────────────────────────────────────────────────────────────────────────────
# Models — NOT baked into the image. Pulled at runtime via RunPod's Model
# Cache feature (HF model ID: techwavelaps/qwen-ai-inpainter), then linked
# into ComfyUI's expected paths by /link_models.sh at boot.
#
# This keeps the image small (~3 GB instead of 25 GB):
#   - Faster GHCR builds (~3 min instead of 17 min)
#   - Faster image pulls on cold workers (less throttle/retry surface)
#   - Models live ONCE per host (overlay-mounted from RunPod's cache)
#     instead of duplicating in every worker's image layer
#
# Required endpoint setting on RunPod:
#   Endpoint > Model Caching > techwavelaps/qwen-ai-inpainter
# ─────────────────────────────────────────────────────────────────────────────

# Ensure model directories exist so the symlinks have parents to land in.
RUN mkdir -p /comfyui/models/text_encoders \
             /comfyui/models/diffusion_models \
             /comfyui/models/vae \
             /comfyui/models/loras

# Symlink script invoked by start.sh before warmups run.
COPY link_models.sh /link_models.sh
RUN chmod +x /link_models.sh

# ─────────────────────────────────────────────────────────────────────────────
# ReActor models — inswapper + buffalo_l + face detection weights
# ─────────────────────────────────────────────────────────────────────────────

# InsightFace buffalo_l face detector/recognizer (~275 MB)
# Use python -m zipfile (stdlib) instead of `unzip` to avoid an apt install.
RUN comfy model download \
        --url https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
        --relative-path models/insightface/models --filename buffalo_l.zip && \
    python -m zipfile -e /comfyui/models/insightface/models/buffalo_l.zip \
        /comfyui/models/insightface/models/buffalo_l && \
    rm /comfyui/models/insightface/models/buffalo_l.zip

# Inswapper face swap model (~550 MB)
RUN comfy model download \
        --url https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx \
        --relative-path models/insightface --filename inswapper_128.onnx

# facexlib + GFPGAN face detection weights
# GFPGAN (used by ReActor face restoration) looks in models/facedetection/, not models/facexlib/
# parsing_parsenet.pth is the parsing model GFPGAN actually loads (parsing_bisenet is
# the BiSeNet variant — ReActor+GFPGAN doesn't use it, so we skip the download).
# Also mirror into /gfpgan/weights/ — the gfpgan library's default cache dir,
# which it will runtime-download into otherwise (adds ~1s + 100MB egress per cold start).
RUN comfy model download \
        --url https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
        --relative-path models/facexlib --filename detection_Resnet50_Final.pth && \
    mkdir -p /comfyui/models/facedetection /gfpgan/weights && \
    cp /comfyui/models/facexlib/detection_Resnet50_Final.pth /comfyui/models/facedetection/ && \
    cp /comfyui/models/facexlib/detection_Resnet50_Final.pth /gfpgan/weights/ && \
    wget -q -O /comfyui/models/facedetection/parsing_parsenet.pth \
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth" && \
    cp /comfyui/models/facedetection/parsing_parsenet.pth /gfpgan/weights/

# GFPGAN face restoration model (~350 MB) — used by ReActor to sharpen swapped faces
RUN comfy model download \
        --url https://huggingface.co/gmk123/GFPGAN/resolve/main/GFPGANv1.4.pth \
        --relative-path models/facerestore_models --filename GFPGANv1.4.pth

# ─────────────────────────────────────────────────────────────────────────────
# Patches — P6: parallel prefetch monkey-patch on core ComfyUI loaders
# ─────────────────────────────────────────────────────────────────────────────
COPY patch_qwen.sh /tmp/patch_qwen.sh
RUN chmod +x /tmp/patch_qwen.sh && /tmp/patch_qwen.sh && rm /tmp/patch_qwen.sh

# Pre-generate matplotlib font cache (avoids ~0.5s rebuild on cold start)
RUN python3 -c "from matplotlib.font_manager import FontManager; FontManager()" || true

# ─────────────────────────────────────────────────────────────────────────────
# Warmup + start scripts
# ─────────────────────────────────────────────────────────────────────────────
COPY warmup_gfpgan.py warmup_qwen.py warmup_insightface.py /
COPY start.sh /start.sh
RUN chmod +x /start.sh
