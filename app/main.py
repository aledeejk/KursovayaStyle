import io
import base64
import os
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.detection import detect_clothes, draw_detections
from app.recommendations import generate_outfit
from app.llm_helper import generate_style_advice

USE_CLIP: bool = os.getenv("USE_CLIP", "false").lower() in ("1", "true", "yes")


app = FastAPI(
    title="Fashion AI – Outfit Recommender",
    description="Детекция одежды (YOLOv8) + CLIP-эмбеддинги + генерация образов",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ItemInfo(BaseModel):
    class_name: str
    confidence: float
    bbox: List[int]
    embedding_norm: float


class RecommendResponse(BaseModel):
    detected_items: List[str]
    present_categories: List[str]
    missing_categories: List[str]
    suggestions: dict
    advice: str
    color_tips: List[str]
    llm_advice: str
    annotated_image_b64: Optional[str] = None
    items_detail: List[ItemInfo] = []


@app.get("/")
async def root():
    return {"message": "Fashion AI API v2.0 — POST /recommend to analyse clothing"}


VALID_ITEM_TYPES = {
    "top", "bottom", "dress", "outerwear", "shoes", "bag", "accessory",
    "t-shirt", "shirt", "sweater", "blouse", "hoodie", "vest",
    "jeans", "trousers", "shorts", "skirt", "leggings", "pants",
    "jacket", "coat", "blazer", "suit",
    "sneakers", "boots", "sandals", "loafers", "heel",
    "backpack", "handbag",
    "hat", "scarf", "belt",
}


@app.post("/recommend", response_model=RecommendResponse)
async def recommend(
    file: UploadFile = File(..., description="Изображение (jpg/png)"),
    style: str = Query("casual", description="casual | formal | sporty"),
    return_image: bool = Query(True, description="Вернуть аннотированное изображение в base64"),
    item_type: Optional[str] = Query(
        None,
        description="Ручной выбор типа предмета (fallback если авто-детекция пустая): t-shirt | shirt | sweater | hoodie | jacket | coat | jeans | trousers | shorts | dress | sneakers | boots",
    ),
    randomize: bool = Query(False, description="Случайная вариация советов"),
):
    """
    Принимает изображение, обнаруживает одежду, вычисляет CLIP-эмбеддинги
    и возвращает рекомендации по составлению образа.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image_bgr is None:
        raise HTTPException(status_code=400, detail="Не удалось декодировать изображение")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    manual = item_type.strip().lower() if item_type and item_type.strip() else None
    if manual and manual not in VALID_ITEM_TYPES:
        manual = None

    detections = detect_clothes(image_rgb)
    clothing_names = [d.class_name for d in detections if d.class_name.lower() != "person"]

    if not clothing_names and manual:
        clothing_names = [manual]

    if not clothing_names:
        clothing_names = [manual or "t-shirt"]

    items_detail: List[ItemInfo] = [
        ItemInfo(
            class_name=det.class_name,
            confidence=round(det.confidence, 3),
            bbox=list(det.bbox),
            embedding_norm=0.0,
        )
        for det in detections
        if det.class_name.lower() != "person"
    ]

    outfit = generate_outfit(clothing_names, style=style, randomize=randomize)

    llm_advice = generate_style_advice(
        detected_items=clothing_names if clothing_names else (detected_names),
        suggestions=outfit["suggestions"],
        style=style,
    )

    annotated_b64 = None
    if return_image:
        annotated = draw_detections(image_rgb, detections)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")

    return RecommendResponse(
        detected_items=outfit["detected"],
        present_categories=outfit["present"],
        missing_categories=outfit["missing"],
        suggestions=outfit["suggestions"],
        advice=outfit["advice"],
        color_tips=outfit["color_tips"],
        llm_advice=llm_advice,
        annotated_image_b64=annotated_b64,
        items_detail=items_detail,
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
