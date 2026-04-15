"""
Pre-warm the Qwen Image Edit pipeline on GPU.

First real request suffers ~10-12s of CPU→VRAM streaming for the 7GB fp8 UNet
+ Lightning + HRP LoRA + CLIP + VAE, plus CUDA kernel compilation. Running a
1-step dummy workflow at container boot shifts that cost off the first real
request by leaving everything already loaded on GPU in ComfyUI's cache.

Runs as a background process during container boot, same pattern as
warmup_gfpgan.py. Waits for the ComfyUI HTTP API to come up, writes a tiny
placeholder image into /comfyui/input/, submits a minimal workflow, polls
/history/ until it reports done, logs the elapsed wall time.
"""
import io
import json
import logging
import os
import time
import urllib.error
import urllib.request

logging.basicConfig(level=logging.INFO, format='[warmup-qwen] %(message)s')
log = logging.getLogger(__name__)

COMFY_URL = os.environ.get("WARMUP_COMFY_URL", "http://127.0.0.1:8188")
INPUT_DIR = "/comfyui/input"
DUMMY_IMAGE = "warmup_qwen.png"
API_WAIT_TIMEOUT = 180
WORKFLOW_TIMEOUT = 180


def wait_for_api(timeout=API_WAIT_TIMEOUT):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/system_stats", timeout=2) as r:
                r.read()
            return True
        except Exception:
            time.sleep(0.5)
    return False


def write_dummy_image():
    from PIL import Image
    os.makedirs(INPUT_DIR, exist_ok=True)
    path = os.path.join(INPUT_DIR, DUMMY_IMAGE)
    Image.new("RGB", (64, 64), (32, 32, 32)).save(path, "PNG")
    return path


def build_workflow():
    return {
        "37": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}},
        "38": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_2.5_vl_7b_fp8_scaled.safetensors", "type": "qwen_image", "device": "default"}},
        "39": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "89": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors", "strength_model": 1, "model": ["37", 0]}},
        "300": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "HRP_20.safetensors", "strength_model": 1.0, "model": ["89", 0]}},
        "66": {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 3, "model": ["300", 0]}},
        "75": {"class_type": "CFGNorm", "inputs": {"strength": 1, "model": ["66", 0]}},
        "78": {"class_type": "LoadImage", "inputs": {"image": DUMMY_IMAGE}},
        "88": {"class_type": "VAEEncode", "inputs": {"pixels": ["78", 0], "vae": ["39", 0]}},
        "111": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"prompt": "warmup", "clip": ["38", 0], "vae": ["39", 0], "image1": ["78", 0]}},
        "112": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["111", 0]}},
        "3": {"class_type": "KSampler", "inputs": {"seed": 0, "steps": 1, "cfg": 1, "sampler_name": "euler", "scheduler": "simple", "denoise": 1, "model": ["75", 0], "positive": ["111", 0], "negative": ["112", 0], "latent_image": ["88", 0]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["39", 0]}},
    }


def submit_and_wait(workflow):
    body = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        prompt_id = json.loads(r.read())["prompt_id"]

    t0 = time.time()
    while time.time() - t0 < WORKFLOW_TIMEOUT:
        try:
            with urllib.request.urlopen(f"{COMFY_URL}/history/{prompt_id}", timeout=5) as r:
                hist = json.loads(r.read())
            if prompt_id in hist:
                status = hist[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"workflow errored: {status}")
                if status.get("completed"):
                    return
        except urllib.error.URLError:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"workflow {prompt_id} did not complete within {WORKFLOW_TIMEOUT}s")


def main():
    t_start = time.time()
    log.info("Waiting for ComfyUI API...")
    if not wait_for_api():
        log.warning(f"ComfyUI API not ready after {API_WAIT_TIMEOUT}s, skipping warmup")
        return
    log.info(f"ComfyUI ready in {time.time()-t_start:.1f}s")

    try:
        write_dummy_image()
        workflow = build_workflow()
        log.info("Submitting 1-step dummy Qwen Edit workflow...")
        t_wf = time.time()
        submit_and_wait(workflow)
        log.info(f"Qwen pipeline warmed up in {time.time()-t_wf:.1f}s "
                 f"(total {time.time()-t_start:.1f}s from boot)")
    except Exception as e:
        log.warning(f"Qwen warmup failed (non-fatal): {e}")


if __name__ == "__main__":
    main()
