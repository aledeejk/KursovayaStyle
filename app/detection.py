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
    print(f"[detection] WARNING: YOLO_MODEL='{requested}' not found. Falling back to '{_FALLBACK_MODEL}'.")
    return _FALLBACK_MODEL

YOLO_MODEL: str = _resolve_yolo_model()

PERSON_CONF: float = float(os.getenv("PERSON_CONF_MIN", "0.40"))
WHOLE_IMAGE_CONF: float = 0.20
ZONE_CONF: float = 0.25
MIN_BOX_AREA: int = 3000
IOU_THRESH: float = 0.65
ACCESSORY_YOLO_CONF: float = 0.20
FULLBODY_ASPECT_RATIO: float = 1.6

COCO_ACCESSORY_CLASSES: Dict[int, str] = {
    24: "backpack", 25: "hat", 26: "handbag", 27: "tie", 28: "suitcase",
}

ZONE_DEFINITIONS: List[Tuple[str, float, float]] = [
    ("head", 0.00, 0.13),
    ("upper_body", 0.10, 0.52),
    ("lower_body", 0.45, 0.85),
    ("feet", 0.80, 1.00),
]

CLIP_LABELS: List[str] = [
    "t-shirt", "shirt", "sweater", "hoodie", "blouse", "tank_top",
    "jacket", "coat", "blazer", "jeans", "trousers", "shorts", "skirt",
    "dress", "leggings", "sneakers", "boots", "shoes", "sandals",
    "hat", "backpack", "handbag", "hair", "glasses", "sunglasses",
    "earrings", "necklace", "bracelet", "watch", "bra", "panties", "briefs"
]

CLIP_PROMPTS: Dict[str, List[str]] = {
    "t-shirt": ["a plain t-shirt", "a white t-shirt", "person wearing a t-shirt"],
    "shirt": ["a dress shirt", "a button-up shirt", "person in a shirt"],
    "sweater": ["a knit sweater", "a woolen sweater", "person in a sweater"],
    "hoodie": ["a hoodie sweatshirt with hood", "a pullover hoodie", "hooded sweatshirt"],
    "blouse": ["a women's blouse", "a silk blouse", "person wearing a blouse"],
    "jacket": ["an outerwear jacket", "a zip-up jacket", "a casual jacket", "a lightweight jacket", "wearing a jacket"],
    "coat": ["a long coat", "a winter coat", "person in an overcoat"],
    "blazer": ["a blazer suit jacket", "a formal blazer"],
    "jeans": ["blue jeans", "denim jeans", "person wearing jeans"],
    "trousers": ["dress trousers", "suit trousers", "formal pants"],
    "shorts": ["denim shorts", "sport shorts", "person in shorts"],
    "skirt": ["a skirt", "a mini skirt", "a long skirt"],
    "dress": ["a dress", "a summer dress", "a woman in a dress"],
    "sneakers": ["sneakers with laces", "sport sneakers", "casual running shoes", "athletic footwear"],
    "boots": ["high leather boots", "winter boots", "knee-high boots", "ankle boots"],
    "shoes": ["formal leather shoes", "dress shoes", "oxford shoes", "flat shoes"],
    "sandals": ["open-toe sandals with straps", "flat sandals", "gladiator sandals", "strappy sandals on feet"],
    "hat": ["a baseball cap", "a beanie hat", "a sun hat", "headwear on head"],
    "backpack": ["a backpack on person's back", "school backpack", "hiking backpack with straps", "rucksack on shoulders"],
    "handbag": ["a woman's handbag", "a small purse carried by hand", "clutch purse", "designer handbag", "tote bag held by handle"],
    "hair": ["natural hair", "hairstyle", "person's hair", "bald head"],
    "tank_top": ["a tank top", "a sleeveless shirt", "a muscle shirt", "person wearing a tank top"],
    "leggings": ["black leggings", "tight leggings", "person wearing leggings", "yoga pants"],
    "bra": ["a woman's bra", "a bra, lingerie", "a bralette"],
    "panties": ["women's panties", "underwear panties", "women's briefs"],
    "briefs": ["men's briefs", "men's underwear", "cotton briefs", "boxer briefs"],
    "glasses": ["a person wearing glasses", "eyeglasses on a face", "reading glasses"],
    "sunglasses": ["a person wearing sunglasses", "dark sunglasses on face"],
    "earrings": ["earrings hanging from ears", "stud earrings", "hoop earrings on ears", "gold earrings", "silver earrings"],
    "necklace": ["a chain necklace around neck", "pendant on necklace", "choker necklace", "neck jewelry", "pearls around neck"],
    "bracelet": ["a bracelet around wrist", "cuff bracelet", "charm bracelet", "wrist bangle", "bracelet chain on wrist"],
    "watch": ["a wristwatch", "a watch on wrist", "luxury watch on hand"],
}

ZONE_LABEL_FILTER: Dict[str, List[str]] = {
    "head": [],
    "upper_body": ["t-shirt", "shirt", "sweater", "hoodie", "blouse", "tank_top", "jacket", "coat", "blazer", "dress", "bra"],
    "lower_body": ["jeans", "trousers", "shorts", "skirt", "dress", "leggings", "panties", "briefs"],
    "feet": ["sneakers", "boots", "shoes", "sandals"],
}

HAT_CONF_MIN: float = 0.40
ACCESSORY_CONF: float = 0.28
FACE_ACCESSORY_LABELS: List[str] = ["glasses", "sunglasses", "earrings"]
NECK_ACCESSORY_LABELS: List[str] = ["necklace"]
WRIST_ACCESSORY_LABELS: List[str] = ["bracelet", "watch"]

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
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL)
    return _yolo_model

def _get_clip():
    global _clip_model, _clip_processor, _clip_device
    if _clip_model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_name = "openai/clip-vit-base-patch32"
        try:
            _clip_model = CLIPModel.from_pretrained(_clip_name, local_files_only=True)
            _clip_processor = CLIPProcessor.from_pretrained(_clip_name, local_files_only=True)
        except Exception:
            _clip_model = CLIPModel.from_pretrained(_clip_name)
            _clip_processor = CLIPProcessor.from_pretrained(_clip_name)
        _clip_model.to(_clip_device)
        _clip_model.eval()
    return _clip_model, _clip_processor, _clip_device

def _get_text_embeddings() -> Tuple[np.ndarray, List[str]]:
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

    inputs = processor(text=flat_prompts, return_tensors="pt", padding=True, truncation=True, max_length=77)
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

def _clip_classify(crop_rgb: np.ndarray, allowed_labels: Optional[List[str]] = None) -> Tuple[str, float]:
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

def _nms(items: List[DetectedItem], iou_thresh: float = IOU_THRESH) -> List[DetectedItem]:
    if not items:
        return []
    ACCESSORY_NAMES = {"glasses", "sunglasses", "earrings", "necklace", "bracelet", "watch"}
    sorted_items = sorted(items, key=lambda d: d.confidence, reverse=True)
    kept = []
    for candidate in sorted_items:
        if candidate.class_name in ACCESSORY_NAMES:
            kept.append(candidate)
            continue
        suppressed = any(_iou(candidate.bbox, k.bbox) > iou_thresh for k in kept)
        if not suppressed:
            kept.append(candidate)
    return kept

def _is_fullbody(px1: int, py1: int, px2: int, py2: int) -> bool:
    w = max(px2 - px1, 1)
    h = max(py2 - py1, 1)
    return (h / w) >= FULLBODY_ASPECT_RATIO

def _crop_zone(image: np.ndarray, px1: int, py1: int, px2: int, py2: int, y_frac_start: float, y_frac_end: float, pad: float = 0.12) -> Optional[np.ndarray]:
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

def _classify_head_zone(crop: np.ndarray, bbox: Tuple[int, int, int, int]) -> List[DetectedItem]:
    results = []
    hat_label, hat_conf = _clip_classify(crop, allowed_labels=["hat"])
    hair_label, hair_conf = _clip_classify(crop, allowed_labels=["hair"])
    print(f"  zone head: hat={hat_conf:.3f}, hair={hair_conf:.3f}")
    if hat_conf >= HAT_CONF_MIN and hat_conf > hair_conf:
        results.append(DetectedItem(bbox=bbox, confidence=round(hat_conf, 3), class_id=-1, class_name="hat", cropped_image=crop))
        print(f"  zone head -> hat accepted ({hat_conf:.3f} > hair {hair_conf:.3f})")
    else:
        print(f"  zone head -> hat rejected (hat={hat_conf:.3f}, hair={hair_conf:.3f})")
    face_label, face_conf = _clip_classify(crop, allowed_labels=FACE_ACCESSORY_LABELS)
    print(f"  zone head accessory: {face_label} ({face_conf:.3f})")
    if face_conf >= ACCESSORY_CONF and face_label != "unknown":
        results.append(DetectedItem(bbox=bbox, confidence=round(face_conf, 3), class_id=-1, class_name=face_label, cropped_image=crop))
    return results

def _classify_wrist_accessories(image: np.ndarray, px1: int, py1: int, px2: int, py2: int, y_start: float, y_end: float) -> List[DetectedItem]:
    results = []
    ph = py2 - py1
    pw = px2 - px1
    h_img, w_img = image.shape[:2]
    zone_y1 = max(0, py1 + int(ph * y_start))
    zone_y2 = min(h_img, py1 + int(ph * y_end))
    zone_x1 = max(0, px1)
    zone_x2 = min(w_img, px2)
    strip_w = max(1, int(pw * 0.18))
    for side, x1s, x2s in [("left", zone_x1, zone_x1 + strip_w), ("right", zone_x2 - strip_w, zone_x2)]:
        if x2s <= x1s or zone_y2 <= zone_y1:
            continue
        wrist_crop = image[zone_y1:zone_y2, x1s:x2s].copy()
        if wrist_crop.size == 0:
            continue
        label, conf = _clip_classify(wrist_crop, allowed_labels=WRIST_ACCESSORY_LABELS)
        print(f"  wrist {side}: {label} ({conf:.3f})")
        if conf >= ACCESSORY_CONF and label != "unknown":
            results.append(DetectedItem(bbox=(x1s, zone_y1, x2s, zone_y2), confidence=round(conf, 3), class_id=-1, class_name=label, cropped_image=wrist_crop))
    neck_y2 = max(0, py1 + int(ph * (y_start + 0.12)))
    if neck_y2 > zone_y1:
        neck_crop = image[zone_y1:neck_y2, zone_x1:zone_x2].copy()
        if neck_crop.size > 0:
            label, conf = _clip_classify(neck_crop, allowed_labels=NECK_ACCESSORY_LABELS)
            print(f"  neck: {label} ({conf:.3f})")
            if conf >= ACCESSORY_CONF and label != "unknown":
                results.append(DetectedItem(bbox=(zone_x1, zone_y1, zone_x2, neck_y2), confidence=round(conf, 3), class_id=-1, class_name=label, cropped_image=neck_crop))
    return results

def _zones_from_person(image: np.ndarray, px1: int, py1: int, px2: int, py2: int) -> List[DetectedItem]:
    items = []
    ph = py2 - py1
    h_img, w_img = image.shape[:2]
    full_body = _is_fullbody(px1, py1, px2, py2)
    active_zones = ZONE_DEFINITIONS if full_body else ZONE_DEFINITIONS[1:3]
    seen_labels = set()
    print(f"[detection] Zones pass ({'full' if full_body else 'half'}-body, {len(active_zones)} zones)")
    for zone_name, y_start, y_end in active_zones:
        crop = _crop_zone(image, px1, py1, px2, py2, y_start, y_end)
        if crop is None:
            continue
        ch, cw = crop.shape[:2]
        if ch * cw < MIN_BOX_AREA:
            print(f"  zone {zone_name}: too small ({ch}x{cw})")
            continue
        zy1 = max(0, py1 + int(ph * y_start))
        zy2 = min(h_img, py1 + int(ph * y_end))
        bbox = (px1, zy1, min(w_img, px2), zy2)
        if zone_name == "head":
            for item in _classify_head_zone(crop, bbox):
                if item.class_name not in seen_labels:
                    seen_labels.add(item.class_name)
                    items.append(item)
            continue
        allowed = ZONE_LABEL_FILTER.get(zone_name)
        label, conf = _clip_classify(crop, allowed_labels=allowed)
        print(f"  zone {zone_name}: {label} ({conf:.3f})")
        if conf < ZONE_CONF or label in seen_labels or label == "unknown":
            continue
        seen_labels.add(label)
        items.append(DetectedItem(bbox=bbox, confidence=round(conf, 3), class_id=-1, class_name=label, cropped_image=crop))
        if zone_name == "upper_body":
            for acc in _classify_wrist_accessories(image, px1, py1, px2, py2, y_start, y_end):
                if acc.class_name not in seen_labels:
                    seen_labels.add(acc.class_name)
                    items.append(acc)
    return items

def _classify_whole_image(image: np.ndarray) -> List[DetectedItem]:
    h, w = image.shape[:2]
    import torch
    model, processor, device = _get_clip()
    text_embs, text_labels = _get_text_embeddings()
    pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
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
        if label not in label_scores or float(score) > label_scores[label]:
            label_scores[label] = float(score)
    results = []
    sorted_items = sorted(label_scores.items(), key=lambda x: x[1], reverse=True)
    print(f"[detection] Whole-image candidates: {[(l, round(c, 3)) for l, c in sorted_items[:5]]}")
    EXCLUDE_FROM_WHOLE_IMAGE = {"bra", "panties", "briefs", "hair"}
    MUTUALLY_EXCLUSIVE = {
        "bags": {"backpack", "handbag"},
        "footwear": {"sneakers", "boots", "shoes", "sandals"},
        "face_accessories": {"glasses", "sunglasses"},
        "jewelry": {"earrings", "necklace", "bracelet", "watch"},
    }
    seen_groups = set()
    for label, conf in sorted_items:
        if conf < WHOLE_IMAGE_CONF or label == "unknown":
            continue
        if label in EXCLUDE_FROM_WHOLE_IMAGE:
            continue
        skip_label = False
        for group_name, group_labels in MUTUALLY_EXCLUSIVE.items():
            if label in group_labels:
                if group_name in seen_groups:
                    print(f"  skipping {label} ({conf:.3f}) - {group_name} already taken")
                    skip_label = True
                    break
                seen_groups.add(group_name)
                break
        if skip_label:
            continue
        is_accessory = label in {"glasses", "sunglasses", "earrings", "necklace", "bracelet", "watch", "hat", "backpack", "handbag"}
        min_conf = 0.22 if is_accessory else 0.23
        if conf < min_conf:
            continue
        results.append(DetectedItem(bbox=(0, 0, w, h), confidence=round(conf, 3), class_id=-1, class_name=label, cropped_image=image.copy()))
        if len(results) >= 1:
            break
    print(f"[detection] Whole-image returning {len(results)} item(s): " + ", ".join(f"{r.class_name}({r.confidence:.3f})" for r in results))
    return results

def _box_valid(x1: int, y1: int, x2: int, y2: int) -> bool:
    bw, bh = x2 - x1, y2 - y1
    return bw >= 40 and bh >= 40 and bw * bh >= MIN_BOX_AREA

def detect_clothes(image: np.ndarray, conf_threshold: float = PERSON_CONF, return_crops: bool = True) -> List[DetectedItem]:
    model = _get_yolo()
    results = model(image, conf=conf_threshold, verbose=False)
    detections = []
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
            elif class_id in COCO_ACCESSORY_CLASSES and confidence >= ACCESSORY_YOLO_CONF:
                acc_name = COCO_ACCESSORY_CLASSES[class_id]
                crop = image[y1c:y2c, x1c:x2c].copy() if return_crops else None
                detections.append(DetectedItem(bbox=(x1c, y1c, x2c, y2c), confidence=round(confidence, 3), class_id=class_id, class_name=acc_name, cropped_image=crop))
                print(f"  YOLO accessory: {acc_name} ({confidence:.3f})")
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
    print(f"[detection] Person path returning {len(detections)} item(s): " + ", ".join(f"{d.class_name}({d.confidence:.2f})" for d in detections))
    return detections

def draw_detections(image: np.ndarray, detections: List[DetectedItem]) -> np.ndarray:
    ZONE_COLORS = {
        "hat": (255, 165, 0), "top": (0, 200, 80), "jacket": (0, 140, 255),
        "dress": (220, 0, 220), "bottom": (30, 144, 255), "pants": (30, 144, 255),
        "shoes": (0, 180, 180), "backpack": (200, 100, 0), "handbag": (180, 50, 180),
        "tie": (255, 80, 80), "glasses": (255, 255, 0), "sunglasses": (128, 128, 0),
        "earrings": (255, 192, 203), "necklace": (218, 165, 32), "bracelet": (0, 255, 255),
        "watch": (192, 192, 192), "tank_top": (144, 238, 144), "leggings": (221, 160, 221),
        "sandals": (210, 105, 30), "boots": (139, 69, 19), "bra": (255, 105, 180),
        "panties": (255, 20, 147), "briefs": (255, 69, 0),
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