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

app = FastAPI(title="Fashion AI", version="2.0.0")

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

VALID_TYPES = {
    "t-shirt", "shirt", "sweater", "hoodie", "blouse", "jacket", "coat", "blazer",
    "jeans", "trousers", "shorts", "skirt", "dress", "leggings",
    "sneakers", "boots", "sandals", "loafers",
    "backpack", "handbag", "hat", "scarf", "belt"
}

@app.get("/")
async def root():
    return {"message": "Fashion AI API"}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend(
    file: UploadFile = File(...),
    style: str = Query("casual", regex="^(casual|formal|sporty)$"),
    return_image: bool = Query(True),
    item_type: Optional[str] = Query(None),
    randomize: bool = Query(False),
    gender: str = Query("auto", regex="^(male|female|auto)$"),
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "Не удалось декодировать изображение")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    detections = detect_clothes(image_rgb)
    detected = [d.class_name for d in detections if d.class_name != "person"]
    
    if not detected and item_type and item_type.lower() in VALID_TYPES:
        detected = [item_type.lower()]
    
    if not detected:
        detected = ["t-shirt"]
    
    outfit = generate_outfit(detected, style=style, randomize=randomize, gender=gender)
    
    llm_advice = generate_style_advice(detected, outfit["suggestions"], style)

    annotated_b64 = None
    if return_image:
        annotated = draw_detections(image_rgb, detections)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        annotated_b64 = base64.b64encode(buffer.tobytes()).decode()
    
    return RecommendResponse(
        detected_items=outfit["detected"],
        present_categories=outfit["present"],
        missing_categories=outfit["missing"],
        suggestions=outfit["suggestions"],
        advice=outfit["advice"],
        color_tips=outfit["color_tips"],
        llm_advice=llm_advice,
        annotated_image_b64=annotated_b64,
        items_detail=[
            ItemInfo(
                class_name=d.class_name,
                confidence=round(d.confidence, 3),
                bbox=list(d.bbox),
                embedding_norm=0.0
            ) for d in detections if d.class_name != "person"
        ]
    )

@app.get("/health")
async def health():
    return {"status": "ok"}