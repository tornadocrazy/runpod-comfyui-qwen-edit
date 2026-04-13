from transformers import pipeline
from PIL import Image
import logging

# Dummy SCORE value (some code may reference this constant)
SCORE = 101

# Silence HuggingFace transformer logs
logging.getLogger("transformers").setLevel(logging.ERROR)
logger = logging.getLogger("reactor_sfw")

def nsfw_image(img_path: str, model_path: str) -> bool:
    """
    NSFW check bypassed.
    Always returns False so no filtering or downloads occur.
    """
    logger.info(f"[ReActor-Bypass] Skipping NSFW check for: {img_path}")
    return False
