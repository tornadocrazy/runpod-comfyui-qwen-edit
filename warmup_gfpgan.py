"""
Pre-warm GFPGAN/facexlib CUDA kernels.
First inference triggers ~5s of CUDA kernel compilation.
Running a dummy pass at startup shifts this off the first request.
"""
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='[warmup-gfpgan] %(message)s')
log = logging.getLogger(__name__)

t0 = time.time()

try:
    import torch
    # Patch: torchvision removed functional_tensor in v0.18+; facexlib still imports it
    import torchvision.transforms.functional as _tvf
    import sys
    sys.modules.setdefault('torchvision.transforms.functional_tensor', _tvf)
    from gfpgan import GFPGANer

    # Load GFPGAN with same config ReActor uses
    restorer = GFPGANer(
        model_path='/comfyui/models/facerestore_models/GFPGANv1.4.pth',
        upscale=1,
        arch='clean',
        channel_multiplier=2,
        bg_upsampler=None,
        device='cuda',
    )

    # Dummy 512x512 image to trigger CUDA kernel compilation
    dummy = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    restorer.enhance(dummy, paste_back=True)

    del restorer
    torch.cuda.empty_cache()
    log.info(f'GFPGAN CUDA kernels warmed up in {time.time()-t0:.1f}s')
except Exception as e:
    log.warning(f'GFPGAN warmup failed (non-fatal): {e}')
