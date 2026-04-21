"""
Pre-warm InsightFace (buffalo_l + inswapper) CUDA sessions.
First ReActor face-swap triggers ~30s of ONNX CUDA kernel compilation
for the buffalo_l detection/recognition stack. Running a dummy analyze
at startup shifts this off the first real request.
"""
import logging
import time
import numpy as np

logging.basicConfig(level=logging.INFO, format='[warmup-insightface] %(message)s')
log = logging.getLogger(__name__)

t0 = time.time()

try:
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    app = FaceAnalysis(name='buffalo_l', root='/comfyui/models/insightface', providers=providers)
    app.prepare(ctx_id=0, det_size=(640, 640))
    app.get(np.zeros((640, 640, 3), dtype=np.uint8))

    swapper = get_model('/comfyui/models/insightface/inswapper_128.onnx', providers=providers)

    del app, swapper
    log.info(f'InsightFace CUDA sessions warmed up in {time.time()-t0:.1f}s')
except Exception as e:
    log.warning(f'InsightFace warmup failed (non-fatal): {e}')
