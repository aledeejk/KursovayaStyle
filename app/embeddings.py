import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from typing import Union
from transformers import CLIPProcessor, CLIPModel


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

_model: CLIPModel = None
_processor: CLIPProcessor = None
_device: str = None


def _load_clip():
    global _model, _processor, _device
    if _model is None:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        _model.to(_device)
        _model.eval()


def get_embedding(image: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Compute a L2-normalised CLIP image embedding.

    Args:
        image: RGB numpy array or PIL Image

    Returns:
        1-D numpy array of shape (512,)
    """
    _load_clip()

    if isinstance(image, np.ndarray):
        pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    inputs = _processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        features = _model.get_image_features(**inputs)
        features = F.normalize(features, p=2, dim=-1)

    return features.cpu().numpy()[0]


def get_text_embedding(text: str) -> np.ndarray:
    """
    Compute a L2-normalised CLIP text embedding.

    Args:
        text: input string

    Returns:
        1-D numpy array of shape (512,)
    """
    _load_clip()

    inputs = _processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77,
    )
    inputs = {k: v.to(_device) for k, v in inputs.items()}

    with torch.no_grad():
        features = _model.get_text_features(**inputs)
        features = F.normalize(features, p=2, dim=-1)

    return features.cpu().numpy()[0]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))
