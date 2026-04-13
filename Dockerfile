# syntax=docker/dockerfile:1.6
# Qwen Image Edit on runpod-worker-comfyui base
# Stable WebSocket handler, parallel model warmup, highvram mode
FROM runpod/worker-comfyui:5.4.1-base

# ─────────────────────────────────────────────────────────────────────────────
# System packages
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

# ─────────────────────────────────────────────────────────────────────────────
# Custom nodes required by Qwen Edit workflows
# ─────────────────────────────────────────────────────────────────────────────
# KJNodes provides: ImageScaleToTotalPixels, CFGNorm, ModelSamplingAuraFlow
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install --no-cache-dir -r requirements.txt

# Remove unused nodes to speed up startup
RUN rm -rf /comfyui/comfy_api_nodes

# ─────────────────────────────────────────────────────────────────────────────
# Models — each in a separate layer for Docker cache efficiency
# ─────────────────────────────────────────────────────────────────────────────

# Qwen 2.5 VL 7B text encoder — fp8 (~7 GB)
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
        --relative-path models/text_encoders \
        --filename qwen_2.5_vl_7b_fp8_scaled.safetensors

# Qwen Image VAE (~few hundred MB)
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors \
        --relative-path models/vae \
        --filename qwen_image_vae.safetensors

# Lightning LoRA — 4-step fast inference (bf16)
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
        --relative-path models/loras \
        --filename Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors

# Qwen Image Edit diffusion model — fp8 mixed (~7-10 GB)
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
        --relative-path models/diffusion_models \
        --filename qwen_image_edit_2511_fp8mixed.safetensors

# ─────────────────────────────────────────────────────────────────────────────
# Warmup + start scripts
# ─────────────────────────────────────────────────────────────────────────────
COPY warmup_models.py /warmup_models.py
COPY start.sh /start.sh
RUN chmod +x /start.sh
