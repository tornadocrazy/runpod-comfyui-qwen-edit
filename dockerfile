FROM runpod/worker-comfyui:5.4.1-base

# ─────────────────────────────────────────────────────────────────────────────
# System packages
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget unzip && \
    rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Python dependencies for ReActor face swap
# Pre-built insightface wheel avoids C compilation issues
# ─────────────────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    https://huggingface.co/iwr-redmond/linux-wheels/resolve/main/insightface-0.7.3-cp312-cp312-linux_x86_64.whl \
    onnxruntime-gpu==1.20.0 \
    facexlib \
    gfpgan

# ─────────────────────────────────────────────────────────────────────────────
# Upgrade ComfyUI — base image ships v0.3.x which lacks
# TextEncodeQwenImageEditPlus (added 2025-09-22 in comfy_extras/nodes_qwen.py).
# Pinned for reproducibility to commit from 2026-02-20.
# ─────────────────────────────────────────────────────────────────────────────
RUN cd /comfyui && \
    git fetch --depth 1 origin 4d172e9ad7c50d08f21df48b04cca9b5f551d0e7 && \
    git checkout 4d172e9ad7c50d08f21df48b04cca9b5f551d0e7 && \
    pip install --no-cache-dir -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Custom nodes required by Qwen Edit workflows
# ─────────────────────────────────────────────────────────────────────────────
# KJNodes provides: ImageScaleToTotalPixels, CFGNorm, ModelSamplingAuraFlow
# Inpaint-CropAndStitch forces inpaint to fit mask bounds (crops + rescales so
# the masked region fills the model's canvas — fixes Qwen hands-cropped bug).
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && pip install --no-cache-dir -r requirements.txt && \
    cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch

# ReActor face swap — comfy-node-install re-installs onnxruntime (CPU), so we
# force-reinstall the GPU variant afterwards
RUN comfy-node-install comfyui-reactor && \
    pip uninstall -y onnxruntime && \
    pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.20.0

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
RUN wget -q --show-progress \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
    -O /comfyui/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors

# Qwen Image VAE
RUN wget -q --show-progress \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors \
    -O /comfyui/models/vae/qwen_image_vae.safetensors

# Lightning LoRA — 4-step fast inference (bf16)
RUN wget -q --show-progress \
    https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    -O /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors

# Hyper-Realistic Portrait identity LoRA — rank 20 (~225 MB)
# Enables strict face-identity preservation when transforming image1 subject.
RUN wget -q --show-progress \
    https://huggingface.co/prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait/resolve/main/HRP_20.safetensors \
    -O /comfyui/models/loras/HRP_20.safetensors

# Qwen Image Edit Inpaint LoRA (ostris) — trained to fill black-painted holes,
# respects mask bounds by design (~590 MB).
RUN wget -q --show-progress \
    https://huggingface.co/ostris/qwen_image_edit_inpainting/resolve/main/qwen_image_edit_inpainting.safetensors \
    -O /comfyui/models/loras/qwen_image_edit_inpainting.safetensors

# Qwen Image Edit diffusion model — fp8 mixed (~7-10 GB)
RUN wget -q --show-progress \
    https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
    -O /comfyui/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors

# ─────────────────────────────────────────────────────────────────────────────
# ReActor models — inswapper + buffalo_l + face detection weights
# ─────────────────────────────────────────────────────────────────────────────

# InsightFace buffalo_l face detector/recognizer (~275 MB)
RUN comfy model download \
        --url https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip \
        --relative-path models/insightface/models --filename buffalo_l.zip && \
    unzip /comfyui/models/insightface/models/buffalo_l.zip \
          -d /comfyui/models/insightface/models/buffalo_l && \
    rm /comfyui/models/insightface/models/buffalo_l.zip

# Inswapper face swap model (~550 MB)
RUN comfy model download \
        --url https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx \
        --relative-path models/insightface --filename inswapper_128.onnx

# facexlib + GFPGAN face detection weights
# GFPGAN (used by ReActor face restoration) looks in models/facedetection/, not models/facexlib/
RUN comfy model download \
        --url https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth \
        --relative-path models/facexlib --filename detection_Resnet50_Final.pth && \
    comfy model download \
        --url https://github.com/xinntao/facexlib/releases/download/v0.2.0/parsing_bisenet.pth \
        --relative-path models/facexlib --filename parsing_bisenet.pth && \
    mkdir -p /comfyui/models/facedetection && \
    cp /comfyui/models/facexlib/detection_Resnet50_Final.pth /comfyui/models/facedetection/ && \
    wget -q -O /comfyui/models/facedetection/parsing_parsenet.pth \
        "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth"

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
COPY warmup_models.py /warmup_models.py
COPY warmup_gfpgan.py /warmup_gfpgan.py
COPY warmup_qwen.py /warmup_qwen.py
COPY start.sh /start.sh
RUN chmod +x /start.sh
