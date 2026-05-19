"""
Clothing detection pipeline (CLIP-based):

Two execution paths:
  A) No person found → whole-image CLIP top-1 → returns 0 or 1 item.
  B) Person found    → anatomical zones → CLIP per zone → NMS → returns 0..4 items.

Set YOLO_MODEL env var to override the YOLO weights file.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


_FALLBACK_MODEL = "yolov8n.pt"


def _resolve_yolo_model() -> str:
    requested = os.getenv("YOLO_MODEL", _FALLBACK_MODEL)
    if requested == _FALLBACK_MODEL:
        return requested
    if os.path.isfile(requested):
        return requested
    print(
        f"[detection] WARNING: YOLO_MODEL='{requested}' not found. "
        f"Falling back to '{_FALLBACK_MODEL}'."
    )
    return _FALLBACK_MODEL


YOLO_MODEL: str = _resolve_yolo_model()

PERSON_CONF: float = float(os.getenv("PERSON_CONF_MIN", "0.40"))
WHOLE_IMAGE_CONF: float = 0.25
ZONE_CONF: float = 0.25
MIN_BOX_AREA: int = 5000
IOU_THRESH: float = 0.50

FULLBODY_ASPECT_RATIO: float = 1.6

COCO_ACCESSORY_CLASSES: Dict[int, str] = {
    24: "backpack", 25: "hat", 26: "handbag", 27: "tie", 28: "suitcase",
}

ZONE_DEFINITIONS: List[Tuple[str, float, float]] = [
    ("head",        0.00, 0.13),
    ("upper_body",  0.10, 0.52),
    ("lower_body",  0.45, 0.85),
    ("feet",        0.80, 1.00),
]

CLIP_LABELS: List[str] = [
    "t-shirt", "shirt", "sweater", "hoodie", "blouse",
    "jacket", "coat", "blazer",
    "jeans", "trousers", "shorts", "skirt", "dress",
    "sneakers", "boots", "shoes",
    "hat", "backpack", "handbag",
]

CLIP_PROMPTS: Dict[str, List[str]] = {
    "t-shirt":   ["a plain t-shirt", "a white t-shirt", "person wearing a t-shirt"],
    "shirt":     ["a dress shirt", "a button-up shirt", "person in a shirt"],
    "sweater":   ["a knit sweater", "a woolen sweater", "person in a sweater"],
    "hoodie":    ["a hoodie with hood", "a zip-up hoodie", "person wearing a hoodie"],
    "blouse":    ["a women's blouse", "a silk blouse", "person wearing a blouse"],
    "jacket":    ["a leather jacket", "a denim jacket", "a bomber jacket"],
    "coat":      ["a long coat", "a winter coat", "person in an overcoat"],
    "blazer":    ["a blazer suit jacket", "a formal blazer"],
    "jeans":     ["blue jeans", "denim jeans", "person wearing jeans"],
    "trousers":  ["dress trousers", "suit trousers", "formal pants"],
    "shorts":    ["denim shorts", "sport shorts", "person in shorts"],
    "skirt":     ["a skirt", "a mini skirt", "a long skirt"],
    "dress":     ["a dress", "a summer dress", "a woman in a dress"],
    "sneakers":  ["sneakers", "white sneakers", "running shoes"],
    "boots":     ["leather boots", "ankle boots", "person wearing boots"],
    "shoes":     ["leather shoes", "loafers", "formal shoes on feet"],
    "hat":       ["a baseball cap", "a beanie hat", "person wearing a hat"],
    "backpack":  ["a backpack on back", "person with a backpack"],
    "handbag":   ["a handbag", "a purse", "woman carrying a handbag"],
}

ZONE_LABEL_FILTER: Dict[str, List[str]] = {
    "head":       ["hat"],
    "upper_body": ["t-shirt", "shirt", "sweater", "hoodie", "blouse", "jacket", "coat", "blazer", "dress"],
    "lower_body": ["jeans", "trousers", "shorts", "skirt", "dress"],
    "feet":       ["sneakers", "boots", "shoes"],
}


@dataclass
class DetectedItem:
    bbox: Tuple[int, int, int, int]
    confidence: float
    class_id: int
    class_name: str
    cropped_image: Optional[np.ndarray] = None


_yolo_model = None
_clip_model = None
_clip_processor = None
_clip_device = None
_clip_text_embs_cache: Optional[np.ndarray] = None
_clip_text_labels_cache: Optional[List[str]] = None


def _get_yolo():
    global _yolo_model
    if _yolo_model is None:
        import torch
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _yolo_model.to(device)
    return _yolo_model


def _get_clip():
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_name = "openai/clip-vit-base-patch32"
        _prev_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        try:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            _clip_model = CLIPModel.from_pretrained(_clip_name, local_files_only=True)
            _clip_processor = CLIPProcessor.from_pretrained(_clip_name, local_files_only=True)
        except Exception:
            if _prev_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            _clip_model = CLIPModel.from_pretrained(_clip_name)
            _clip_processor = CLIPProcessor.from_pretrained(_clip_name)
        finally:
            if _prev_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            elif _prev_offline:
                os.environ["TRANSFORMERS_OFFLINE"] = _prev_offline
        _clip_model.to(_clip_device)
        _clip_model.eval()
    return _clip_model, _clip_processor, _clip_device


def _get_text_embeddings() -> Tuple[np.ndarray, List[str]]:
    """Build and cache normalised text embeddings for all CLIP_LABELS x prompts."""
    global _clip_text_embs_cache, _clip_text_labels_cache
    if _clip_text_embs_cache is not None:
        return _clip_text_embs_cache, _clip_text_labels_cache

    import torch
    model, processor, device = _get_clip()

    flat_labels: List[str] = []
    flat_prompts: List[str] = []
    for label in CLIP_LABELS:
        for p in CLIP_PROMPTS.get(label, [f"a photo of {label}"]):
            flat_labels.append(label)
            flat_prompts.append(p)

    inputs = processor(
        text=flat_prompts, return_tensors="pt",
        padding=True, truncation=True, max_length=77,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.get_text_features(**inputs)
    if isinstance(out, torch.Tensor):
        raw = out.float()
    else:
        raw = out.pooler_output.float() if out.pooler_output is not None else out.last_hidden_state[:, 0, :].float()
    raw = raw / raw.norm(dim=-1, keepdim=True)

    _clip_text_embs_cache = raw.cpu().numpy()
    _clip_text_labels_cache = flat_labels
    return _clip_text_embs_cache, _clip_text_labels_cache


def _clip_classify(
    crop_rgb: np.ndarray,
    allowed_labels: Optional[List[str]] = None,
) -> Tuple[str, float]:
    """
    Classify a crop with CLIP against clothing prompts.
    Returns (best_label, cosine_similarity_score).
    If allowed_labels is given, only those labels compete.
    """
    import torch
    try:
        model, processor, device = _get_clip()
        text_embs, text_labels = _get_text_embeddings()

        pil_img = Image.fromarray(crop_rgb.astype(np.uint8)).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.get_image_features(**inputs)
        if isinstance(out, torch.Tensor):
            img_raw = out.float()
        else:
            img_raw = out.pooler_output.float() if out.pooler_output is not None else out.last_hidden_state[:, 0, :].float()
        img_feat = (img_raw / img_raw.norm(dim=-1, keepdim=True)).cpu().numpy()[0]

        sims = text_embs @ img_feat

        label_scores: Dict[str, float] = {}
        for label, score in zip(text_labels, sims):
            if allowed_labels and label not in allowed_labels:
                continue
            if label not in label_scores or float(score) > label_scores[label]:
                label_scores[label] = float(score)

        if not label_scores:
            return "unknown", 0.0
        best_label = max(label_scores, key=label_scores.__getitem__)
        return best_label, label_scores[best_label]

    except Exception as exc:
        print(f"[CLIP classify error] {exc}")
        return "unknown", 0.0


def _iou(a: Tuple[int,int,int,int], b: Tuple[int,int,int,int]) -> float:
    """Intersection over Union for two bboxes (x1,y1,x2,y2)."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _nms(items: List["DetectedItem"], iou_thresh: float = IOU_THRESH) -> List["DetectedItem"]:
    """Greedy NMS: keep highest-confidence item, suppress overlapping ones."""
    if not items:
        return []
    sorted_items = sorted(items, key=lambda d: d.confidence, reverse=True)
    kept: List["DetectedItem"] = []
    for candidate in sorted_items:
        suppressed = any(
            _iou(candidate.bbox, k.bbox) > iou_thresh
            for k in kept
        )
        if not suppressed:
            kept.append(candidate)
    return kept


def _is_fullbody(px1: int, py1: int, px2: int, py2: int) -> bool:
    w = max(px2 - px1, 1)
    h = max(py2 - py1, 1)
    return (h / w) >= FULLBODY_ASPECT_RATIO


def _crop_zone(
    image: np.ndarray,
    px1: int, py1: int, px2: int, py2: int,
    y_frac_start: float, y_frac_end: float,
    pad: float = 0.12,
) -> Optional[np.ndarray]:
    """Crop a zone with slight padding on all sides."""
    ph = py2 - py1
    pw = px2 - px1
    h, w = image.shape[:2]
    zone_h = ph * (y_frac_end - y_frac_start)
    y_pad = int(zone_h * pad)
    x_pad = int(pw * pad)
    zy1 = max(0, py1 + int(ph * y_frac_start) - y_pad)
    zy2 = min(h, py1 + int(ph * y_frac_end) + y_pad)
    x1c = max(0, px1 - x_pad)
    x2c = min(w, px2 + x_pad)
    if zy2 - zy1 < 20 or x2c - x1c < 20:
        return None
    return image[zy1:zy2, x1c:x2c].copy()


def _zones_from_person(
    image: np.ndarray,
    px1: int, py1: int, px2: int, py2: int,
) -> List[DetectedItem]:
    """Split person bbox into anatomical zones; classify each with CLIP."""
    items: List[DetectedItem] = []
    ph = py2 - py1
    h_img, w_img = image.shape[:2]
    full_body = _is_fullbody(px1, py1, px2, py2)
    active_zones = ZONE_DEFINITIONS if full_body else ZONE_DEFINITIONS[1:3]
    seen_labels: set = set()

    print(f"[detection] Zones pass ({'full' if full_body else 'half'}-body, {len(active_zones)} zones)")

    for zone_name, y_start, y_end in active_zones:
        crop = _crop_zone(image, px1, py1, px2, py2, y_start, y_end)
        if crop is None:
            continue
        ch, cw = crop.shape[:2]
        if ch * cw < MIN_BOX_AREA:
            print(f"  zone {zone_name}: too small ({ch}x{cw})")
            continue
        allowed = ZONE_LABEL_FILTER.get(zone_name)
        label, conf = _clip_classify(crop, allowed_labels=allowed)
        print(f"  zone {zone_name}: {label} ({conf:.3f})")
        if conf < ZONE_CONF or label in seen_labels or label == "unknown":
            continue
        zy1 = max(0, py1 + int(ph * y_start))
        zy2 = min(h_img, py1 + int(ph * y_end))
        seen_labels.add(label)
        items.append(DetectedItem(
            bbox=(px1, zy1, min(w_img, px2), zy2),
            confidence=round(conf, 3),
            class_id=-1,
            class_name=label,
            cropped_image=crop,
        ))

    return items


def _classify_whole_image(image: np.ndarray) -> List[DetectedItem]:
    """
    No-person fallback: top-1 CLIP on the whole image.
    Returns exactly 0 or 1 DetectedItem.
    """
    h, w = image.shape[:2]
    label, conf = _clip_classify(image)
    print(f"[detection] Whole-image top-1: {label} ({conf:.3f})")
    if conf < WHOLE_IMAGE_CONF or label == "unknown":
        return []
    return [DetectedItem(
        bbox=(0, 0, w, h),
        confidence=round(conf, 3),
        class_id=-1,
        class_name=label,
        cropped_image=image.copy(),
    )]


def _box_valid(x1: int, y1: int, x2: int, y2: int) -> bool:
    bw, bh = x2 - x1, y2 - y1
    return bw >= 40 and bh >= 40 and bw * bh >= MIN_BOX_AREA


def detect_clothes(
    image: np.ndarray,
    conf_threshold: float = PERSON_CONF,
    return_crops: bool = True,
) -> List[DetectedItem]:
    """
    Path A (no person): whole-image CLIP → top-1 → 0 or 1 item.
    Path B (person found): zones CLIP → NMS → 0..4 items.
    """
    model = _get_yolo()
    results = model(image, conf=conf_threshold, verbose=False)

    detections: List[DetectedItem] = []
    found_person = False
    n_persons = 0
    h_img, w_img = image.shape[:2]

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names.get(class_id, f"class_{class_id}")

            x1c = max(0, x1); y1c = max(0, y1)
            x2c = min(w_img, x2); y2c = min(h_img, y2)

            if not _box_valid(x1c, y1c, x2c, y2c):
                continue

            if class_name == "person" and confidence >= PERSON_CONF:
                n_persons += 1
                found_person = True
                zone_items = _zones_from_person(image, x1c, y1c, x2c, y2c)
                detections.extend(zone_items)

            elif class_id in COCO_ACCESSORY_CLASSES:
                acc_name = COCO_ACCESSORY_CLASSES[class_id]
                crop = image[y1c:y2c, x1c:x2c].copy() if return_crops else None
                detections.append(DetectedItem(
                    bbox=(x1c, y1c, x2c, y2c),
                    confidence=round(confidence, 3),
                    class_id=class_id,
                    class_name=acc_name,
                    cropped_image=crop,
                ))

    print(f"[detection] YOLO persons found: {n_persons}")

    if not found_person:
        whole = _classify_whole_image(image)
        if whole:
            return whole
        print(f"[detection] Empty: whole-image conf below WHOLE_IMAGE_CONF={WHOLE_IMAGE_CONF}")
        return []

    if not detections:
        print("[detection] Empty: person found but all zones below ZONE_CONF")
        return []

    detections = _nms(detections)
    detections.sort(key=lambda d: d.confidence, reverse=True)
    print(f"[detection] Person path returning {len(detections)} item(s): " +
          ", ".join(f"{d.class_name}({d.confidence:.2f})" for d in detections))
    return detections


def draw_detections(image: np.ndarray, detections: List[DetectedItem]) -> np.ndarray:
    ZONE_COLORS = {
        "hat": (255, 165, 0), "top": (0, 200, 80), "jacket": (0, 140, 255),
        "dress": (220, 0, 220), "bottom": (30, 144, 255), "pants": (30, 144, 255),
        "shoes": (0, 180, 180), "backpack": (200, 100, 0),
        "handbag": (180, 50, 180), "tie": (255, 80, 80),
    }
    img = image.copy()
    for det in detections:
        color = ZONE_COLORS.get(det.class_name, (0, 200, 80))
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name}: {det.confidence:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        ly = y1 - 6 if y1 > 20 else y1 + lh + 6
        cv2.rectangle(img, (x1, ly - lh - 4), (x1 + lw, ly + 2), color, -1)
        cv2.putText(img, label, (x1, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img
