#!/usr/bin/env bash
set -e

# Models live in RunPod's HF Model Cache (mounted at /runpod-volume/...) —
# we symlink them into ComfyUI's expected paths via /link_models.sh below
# before any warmups try to read them.
echo "worker-comfyui: Linking models from RunPod Model Cache..."
/link_models.sh
echo "worker-comfyui: Models linked"

# ─────────────────────────────────────────────────────────────────────────────
# Disk-read-early: warm OS page cache for big model files in parallel with
# ComfyUI startup. The cache files mount via overlayfs from host SSD; first
# read is ~17s for the 19 GB UNet (cache mount cold), repeat reads are
# ~2s (page cache hot). Running this in background while ComfyUI boots
# means the warmup workflow can read at RAM speed. Now reads through the
# symlinks created by /link_models.sh.
# ─────────────────────────────────────────────────────────────────────────────
(
    cat /comfyui/models/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors > /dev/null 2>&1 || true
    cat /comfyui/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors > /dev/null 2>&1 || true
    cat /comfyui/models/loras/Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors > /dev/null 2>&1 || true
    cat /comfyui/models/loras/HRP_20.safetensors > /dev/null 2>&1 || true
    cat /comfyui/models/loras/qwen_image_edit_inpainting.safetensors > /dev/null 2>&1 || true
    cat /comfyui/models/vae/qwen_image_vae.safetensors > /dev/null 2>&1 || true
) &
echo "worker-comfyui: disk-read-early started (PID $!) — warming page cache for big models"

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po 'libtcmalloc.so.\d' | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# ComfyUI-Manager removed in dockerfile — no longer need the offline-mode
# command. Was: comfy-manager-set-mode offline

echo "worker-comfyui: Starting ComfyUI (Qwen Edit)"

# COMFY_LOG_LEVEL=INFO suppresses the per-module "lowvram: loaded module
# regularly" + "Backend eager selected for dequantize_per_tensor_fp8" spam
# (~120 lines per cold start) — it's DEBUG noise, not actual lowvram mode.
: "${COMFY_LOG_LEVEL:=INFO}"
# SageAttention v1.0.6 (latest on PyPI) produces all-black output on Qwen
# Image Edit fp8mixed — the int8 quantization compounds with the model's
# fp8 weights and Lightning LoRA, output collapses. v2.x (github-only,
# requires nvcc to build) may fix this. For now, sage is installed but
# NOT enabled by default. Override via COMFY_EXTRA_ARGS env var to test.
#
#   --highvram             keep models in VRAM between requests
#   --disable-smart-memory skip ComfyUI's smart eviction layer; we control
#                          memory via highvram + Lightning's tight VRAM budget
: "${COMFY_EXTRA_ARGS:=--highvram --disable-smart-memory}"

# Wait for ComfyUI's HTTP API to come up (~9s typically). We replace
# warmup_qwen.py here: that script's full 1-step dummy workflow took ~18s
# of cold-boot time but its only durable benefit (CUDA kernel JIT for
# KSampler) is a ~5s saving on jobs 2+. For sporadic-fire pattern where
# workers handle ~1 job per cold-start lifetime, that's a net loss. The
# first real job now pays the kernel JIT + LoRA-chain UNet load itself,
# but we save ~17s on cold-boot regardless of whether jobs follow.
# disk-read-early (above) still warms the OS page cache so the real job's
# UNet load is RAM-speed, not slow disk-speed.
wait_for_comfy_api() {
    echo "worker-comfyui: Waiting for ComfyUI API to come up..."
    for i in $(seq 1 60); do
        if python -c "import urllib.request, json; urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=1).read()" 2>/dev/null; then
            echo "worker-comfyui: ComfyUI API ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "worker-comfyui - WARNING: ComfyUI API did not come up within 60s, starting handler anyway" >&2
    return 1
}

if [ "$SERVE_API_LOCALLY" == "true" ]; then
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --listen --verbose "${COMFY_LOG_LEVEL}" --log-stdout ${COMFY_EXTRA_ARGS} &

    # Pre-warm GFPGAN/facexlib CUDA kernels in background.
    # First GFPGAN inference triggers ~5s CUDA kernel compilation —
    # running a dummy pass at startup shifts this off the first request.
    python -u /warmup_gfpgan.py &

    # Pre-warm InsightFace (buffalo_l + inswapper) ONNX CUDA sessions.
    # First ReActor face-swap triggers ~30s of CUDA kernel compile.
    python -u /warmup_insightface.py &

    wait_for_comfy_api

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py --rp_serve_api --rp_api_host=0.0.0.0
else
    python -u /comfyui/main.py --disable-auto-launch --disable-metadata --verbose "${COMFY_LOG_LEVEL}" --log-stdout ${COMFY_EXTRA_ARGS} &

    # Pre-warm GFPGAN/facexlib CUDA kernels in background.
    python -u /warmup_gfpgan.py &

    # Pre-warm InsightFace (buffalo_l + inswapper) ONNX CUDA sessions.
    python -u /warmup_insightface.py &

    wait_for_comfy_api

    echo "worker-comfyui: Starting RunPod Handler"
    python -u /handler.py
fi
