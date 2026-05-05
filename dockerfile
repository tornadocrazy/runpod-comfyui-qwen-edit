FROM runpod/worker-comfyui:5.8.5-base
# build trigger: 2026-05-05T02:13

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
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes && \
    git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor comfyui-reactor && \
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux && \
    git clone --depth 1 https://github.com/lenML/comfyui_qwen_image_edit_adv

# One uv resolver pass for all node requirements + pin GPU onnxruntime
# After node deps drag in CPU onnxruntime, force a clean reinstall of -gpu so
# insightface gets a working InferenceSession (otherwise CPU/GPU share the
# same `onnxruntime/` package dir and the uninstall corrupts the namespace).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt \
        -r /comfyui/custom_nodes/comfyui-reactor/requirements.txt \
        -r /comfyui/custom_nodes/comfyui_controlnet_aux/requirements.txt && \
    uv pip uninstall onnxruntime onnxruntime-gpu && \
    uv pip install --reinstall onnxruntime-gpu==1.22.0 && \
    python -c "import onnxruntime; assert hasattr(onnxruntime, 'InferenceSession'), 'onnxruntime broken'; print('onnxruntime OK:', onnxruntime.__version__)"

# Bypass ReActor NSFW filter — avoids downloading large classifier at runtime
COPY reactor_sfw.py /comfyui/custom_nodes/comfyui-reactor/scripts/reactor_sfw.py

# Remove unused nodes to speed up startup
RUN rm -rf /comfyui/comfy_api_nodes

# ─────────────────────────────────────────────────────────────────────────────
# Models — plain wget, one per layer for Docker cache
# ─────────────────────────────────────────────────────────────────────────────

# Ensure model directories exist (some may not be in base image)
RUN mkdir -p /comfyui/models/text_encoders \
             /comfyui/models/diffusion_models \
             /comfyui/models/vae \
             /comfyui/models/loras

# Qwen 2.5 VL 7B text encoder — fp8 (~7 GB)
RUN hf download Comfy-Org/Qwen-Image_ComfyUI \
        split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
        --local-dir /tmp/hf && \
    mv /tmp/hf/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
       /comfyui/models/text_encoders/ && \
    rm -rf /tmp/hf

# Qwen Image VAE
RUN wget -q \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors \
    -O /comfyui/models/vae/qwen_image_vae.safetensors

# Lightning LoRA — 4-step fast inference (bf16) (~1.5 GB)
RUN wget -q \
    https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    -O /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors

# Hyper-Realistic Portrait identity LoRA — rank 20 (~225 MB)
RUN wget -q \
    https://huggingface.co/prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait/resolve/main/HRP_20.safetensors \
    -O /comfyui/models/loras/HRP_20.safetensors

# Union ControlNet LoRA (~944 MB) — DWPose / OpenPose conditioning
RUN wget -q \
    https://huggingface.co/Comfy-Org/Qwen-Image-DiffSynth-ControlNets/resolve/main/split_files/loras/qwen_image_union_diffsynth_lora.safetensors \
    -O /comfyui/models/loras/qwen_image_union_diffsynth_lora.safetensors

# Fusion LoRA (~236 MB) — blends subject into background naturally
RUN wget -q \
    https://huggingface.co/vantagewithai/Qwen_Image_Edit_LoRAs/resolve/main/qwen_image_edit_fusion.safetensors \
    -O /comfyui/models/loras/qwen_image_edit_fusion.safetensors

# Qwen Image Edit diffusion model — fp8 mixed (~14 GB) — biggest file, biggest win from hf_transfer
RUN hf download Comfy-Org/Qwen-Image-Edit_ComfyUI \
        split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
        --local-dir /tmp/hf && \
    mv /tmp/hf/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
       /comfyui/models/diffusion_models/ && \
    rm -rf /tmp/hf

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
