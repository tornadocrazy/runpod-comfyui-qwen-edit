"""
ReActor source-face pre-analysis (parallel with KSampler) — CPU only.

Hooks ComfyUI's POST /prompt endpoint via aiohttp middleware. When a prompt
arrives that contains a ReActorFaceSwap node, this fires off a background
thread that runs InsightFace face detection on the source image and populates
ReActor's module-level cache (SOURCE_FACES + SOURCE_IMAGE_HASH).

CRITICAL: the analyzer here uses CPUExecutionProvider, NOT CUDA. PyTorch's
KSampler internally uses CUDA graph capture; concurrent GPU operations from
other threads break the capture with cudaErrorStreamCaptureUnsupported.
Running the analyze on CPU sidesteps the conflict entirely — KSampler keeps
the GPU, our pre-analysis works the CPU in parallel, no shared state.

By the time KSampler + VAEDecode finish (~12s) and ReActorFaceSwap fires,
the analysis (~3-5s on CPU) is already done and sitting in cache → ReActor's
built-in MD5-hash check hits → swap proceeds without re-analyzing. The
actual GPU swap (INSwapper) inside ReActorFaceSwap continues to use CUDA
as before.

Net: ~3-4s saved per job. No workflow changes needed.
"""
import os
import sys
import threading
import traceback

# Lazy-initialized CPU-only InsightFace analyzer. Created on first job use,
# reused thereafter. Holds buffalo_l (det + landmarks + embedding) on CPU.
_cpu_analyzer = None
_cpu_analyzer_lock = threading.Lock()


def _get_cpu_analyzer():
    """Build (once) and return a CPU-only ReActorFaceAnalysis."""
    global _cpu_analyzer
    if _cpu_analyzer is not None:
        return _cpu_analyzer
    with _cpu_analyzer_lock:
        if _cpu_analyzer is not None:
            return _cpu_analyzer
        reactor_dir = "/comfyui/custom_nodes/comfyui-reactor"
        if reactor_dir not in sys.path:
            sys.path.insert(0, reactor_dir)
        from reactor_core.analyzer import ReActorFaceAnalysis
        import folder_paths

        insightface_path = os.path.join(folder_paths.models_dir, "insightface")
        analyzer = ReActorFaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
            root=insightface_path,
        )
        analyzer.prepare(ctx_id=-1, det_size=(640, 640))
        _cpu_analyzer = analyzer
        print("[reactor_preanalyze] CPU analyzer initialized (buffalo_l)", flush=True)
        return _cpu_analyzer


def _resolve_source_filename(prompt_data):
    """Walk the prompt graph to find ReActorFaceSwap → LoadImage source filename."""
    if not isinstance(prompt_data, dict):
        return None
    for node in prompt_data.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") != "ReActorFaceSwap":
            continue
        src_ref = node.get("inputs", {}).get("source_image")
        if not isinstance(src_ref, list) or len(src_ref) < 1:
            continue
        src_node = prompt_data.get(str(src_ref[0]))
        if isinstance(src_node, dict) and src_node.get("class_type") == "LoadImage":
            return src_node.get("inputs", {}).get("image")
    return None


def _do_preanalyze(image_filename):
    """Run InsightFace on source image; populate ReActor's cache globals."""
    try:
        # Lazy imports — ReActor's modules aren't on sys.path until ComfyUI loads custom_nodes
        reactor_dir = "/comfyui/custom_nodes/comfyui-reactor"
        if reactor_dir not in sys.path:
            sys.path.insert(0, reactor_dir)
        from scripts import reactor_swapper
        from reactor_utils import get_image_md5hash

        import folder_paths
        import cv2
        import numpy as np
        from PIL import Image

        input_dir = folder_paths.get_input_directory()
        image_path = os.path.join(input_dir, image_filename)
        if not os.path.exists(image_path):
            print(f"[reactor_preanalyze] source image not found: {image_path}", flush=True)
            return

        img_pil = Image.open(image_path).convert("RGB")
        img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

        new_hash = get_image_md5hash(img_bgr)
        if (
            reactor_swapper.SOURCE_IMAGE_HASH == new_hash
            and reactor_swapper.SOURCE_FACES is not None
        ):
            print(f"[reactor_preanalyze] cache hit for {image_filename} — skipping", flush=True)
            return

        print(f"[reactor_preanalyze] analyzing {image_filename} on CPU (parallel with KSampler)...", flush=True)
        analyzer = _get_cpu_analyzer()
        faces = analyzer.get(img_bgr)
        # If detection failed at 640, retry at 320 (matches ReActor's analyze_faces fallback)
        if not faces:
            analyzer.prepare(ctx_id=-1, det_size=(320, 320))
            faces = analyzer.get(img_bgr)
            analyzer.prepare(ctx_id=-1, det_size=(640, 640))
        reactor_swapper.SOURCE_IMAGE_HASH = new_hash
        reactor_swapper.SOURCE_FACES = faces
        print(f"[reactor_preanalyze] cache populated for {image_filename} ({len(faces) if faces else 0} face(s))", flush=True)
    except Exception as e:
        print(f"[reactor_preanalyze] error: {e}", flush=True)
        traceback.print_exc()


def _register_middleware():
    """Attach aiohttp middleware to ComfyUI's PromptServer."""
    try:
        from aiohttp import web
        from server import PromptServer

        @web.middleware
        async def preanalyze_middleware(request, handler):
            if request.method == "POST" and request.path in ("/prompt", "/api/prompt"):
                try:
                    data = await request.json()
                    prompt = data.get("prompt") if isinstance(data, dict) else None
                    src_filename = _resolve_source_filename(prompt)
                    if src_filename:
                        t = threading.Thread(
                            target=_do_preanalyze,
                            args=(src_filename,),
                            daemon=True,
                            name="reactor-preanalyze",
                        )
                        t.start()
                        print(
                            f"[reactor_preanalyze] kicked off thread for {src_filename}",
                            flush=True,
                        )
                except Exception as e:
                    print(f"[reactor_preanalyze] middleware peek failed: {e}", flush=True)
            return await handler(request)

        PromptServer.instance.app.middlewares.append(preanalyze_middleware)
        print("[reactor_preanalyze] middleware registered on POST /prompt", flush=True)
    except Exception as e:
        print(f"[reactor_preanalyze] failed to register: {e}", flush=True)
        traceback.print_exc()


_register_middleware()

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
