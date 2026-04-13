FROM runpod/worker-comfyui:5.4.1-base

# ─────────────────────────────────────────────────────────────────────────────
# System packages
# ─────────────────────────────────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

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
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes && \
    cd ComfyUI-KJNodes && \
    pip install --no-cache-dir -r requirements.txt

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

# Qwen Image Edit diffusion model — fp8 mixed (~7-10 GB)
RUN wget -q --show-progress \
    https://huggingface.co/Comfy-Org/Qwen-Image-Edit_ComfyUI/resolve/main/split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors \
    -O /comfyui/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors

# ─────────────────────────────────────────────────────────────────────────────
# Warmup + start scripts
# ─────────────────────────────────────────────────────────────────────────────
COPY warmup_models.py /warmup_models.py
COPY start.sh /start.sh
RUN chmod +x /start.sh
