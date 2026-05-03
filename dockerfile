FROM runpod/worker-comfyui:5.8.5-base
# build trigger: 2026-05-04T01:00

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
    onnxruntime-gpu==1.22.0 \
    facexlib \
    gfpgan

# ─────────────────────────────────────────────────────────────────────────────
# Custom nodes required by Qwen Edit workflows
# ─────────────────────────────────────────────────────────────────────────────
# Clone all nodes in one step (skips comfy-node-install registry overhead)
RUN cd /comfyui/custom_nodes && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes && \
    git clone --depth 1 https://github.com/lquesada/ComfyUI-Inpaint-CropAndStitch && \
    git clone --depth 1 https://github.com/Gourieff/ComfyUI-ReActor comfyui-reactor && \
    git clone --depth 1 https://github.com/1038lab/ComfyUI-RMBG

# One pip resolver pass for all node requirements + pin GPU onnxruntime
RUN pip install --no-cache-dir \
        -r /comfyui/custom_nodes/ComfyUI-KJNodes/requirements.txt \
        -r /comfyui/custom_nodes/comfyui-reactor/requirements.txt \
        -r /comfyui/custom_nodes/ComfyUI-RMBG/requirements.txt && \
    pip uninstall -y onnxruntime && \
    pip install --no-cache-dir --force-reinstall onnxruntime-gpu==1.22.0

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
RUN wget -q \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors \
    -O /comfyui/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors

# Qwen Image VAE
RUN wget -q \
    https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors \
    -O /comfyui/models/vae/qwen_image_vae.safetensors

# Lightning LoRA — 4-step fast inference (bf16)
RUN wget -q \
    https://huggingface.co/lightx2v/Qwen-Image-Edit-2511-Lightning/resolve/main/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors \
    -O /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors

# Hyper-Realistic Portrait identity LoRA — rank 20 (~225 MB)
RUN wget -q \
    https://huggingface.co/prithivMLmods/Qwen-Image-Edit-2511-Hyper-Realistic-Portrait/resolve/main/HRP_20.safetensors \
    -O /comfyui/models/loras/HRP_20.safetensors

# Qwen Image Edit Inpaint LoRA (~590 MB)
RUN wget -q \
    https://huggingface.co/ostris/qwen_image_edit_inpainting/resolve/main/qwen_image_edit_inpainting.safetensors \
    -O /comfyui/models/loras/qwen_image_edit_inpainting.safetensors

# Qwen Image Union DiffSynth ControlNet LoRA — pose/depth/canny (~944 MB)
RUN wget -q \
    https://huggingface.co/Comfy-Org/Qwen-Image-DiffSynth-ControlNets/resolve/main/split_files/loras/qwen_image_union_diffsynth_lora.safetensors \
    -O /comfyui/models/loras/qwen_image_union_diffsynth_lora.safetensors

# Qwen Image Edit diffusion model — fp8 mixed (~7-10 GB)
RUN wget -q \
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

# RMBG-2.0 (background removal) — node downloads to models/RMBG/RMBG-2.0/ at runtime
# Pre-bake all files so no runtime downloads needed
RUN mkdir -p /comfyui/models/RMBG/RMBG-2.0 && \
    wget -q -O /comfyui/models/RMBG/RMBG-2.0/birefnet.py \
        "https://huggingface.co/1038lab/RMBG-2.0/raw/main/birefnet.py" && \
    wget -q -O /comfyui/models/RMBG/RMBG-2.0/BiRefNet_config.py \
        "https://huggingface.co/1038lab/RMBG-2.0/raw/main/BiRefNet_config.py" && \
    wget -q -O /comfyui/models/RMBG/RMBG-2.0/config.json \
        "https://huggingface.co/1038lab/RMBG-2.0/resolve/main/config.json"
RUN --mount=type=cache,target=/root/.cache \
    comfy model download \
        --url https://huggingface.co/1038lab/RMBG-2.0/resolve/main/model.safetensors \
        --relative-path models/RMBG/RMBG-2.0 --filename model.safetensors

# ─────────────────────────────────────────────────────────────────────────────
# Patches — P6: parallel prefetch monkey-patch on core ComfyUI loaders
# ─────────────────────────────────────────────────────────────────────────────
COPY --chmod=0755 patch_qwen.sh /tmp/patch_qwen.sh
RUN /tmp/patch_qwen.sh && rm /tmp/patch_qwen.sh

# Pre-generate matplotlib font cache (avoids ~0.5s rebuild on cold start)
RUN python3 -c "from matplotlib.font_manager import FontManager; FontManager()" || true

# ─────────────────────────────────────────────────────────────────────────────
# Warmup + start scripts
# ─────────────────────────────────────────────────────────────────────────────
COPY warmup_gfpgan.py warmup_qwen.py warmup_insightface.py /
COPY --chmod=0755 start.sh /start.sh
