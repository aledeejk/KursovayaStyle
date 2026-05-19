"""
Enhanced FastAPI Server for Fashion AI System
"""
import os
import io
import time
import base64
from typing import List, Optional, Dict
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Импорты из проекта
from app.models.fashion_detector import FashionDetector
from app.models.color_analyzer import ColorAnalyzer
from app.services.outfit_generator import RuleBasedGenerator, OutfitStyle, ClothingItem
from app.services.weather_api import WeatherService
from app.services.personalization import PersonalizationService


# === Models ===
class ClothingItemResponse(BaseModel):
    id: str
    category: str
    subcategory: str
    color: str
    color_hex: str
    confidence: float
    bbox: List[int]


class OutfitRecommendation(BaseModel):
    items: List[ClothingItemResponse]
    style: str
    score: float
    reasoning: str


class AnalysisResponse(BaseModel):
    image_id: str
    detected_items: List[ClothingItemResponse]
    outfits: List[OutfitRecommendation]
    weather_info: Optional[Dict] = None
    processing_time_ms: float


# === Global Services ===
fashion_detector = None
color_analyzer = None
outfit_generator = None
weather_service = None
personalization_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global fashion_detector, color_analyzer, outfit_generator, weather_service, personalization_service
    
    print("🚀 Запуск Fashion AI System...")
    
    # Инициализация сервисов
    model_path = os.getenv('FASHION_MODEL', 'yolov8n.pt')
    fashion_detector = FashionDetector(model_path=model_path)
    color_analyzer = ColorAnalyzer(n_colors=3)
    outfit_generator = RuleBasedGenerator()
    weather_service = WeatherService(api_key=os.getenv('WEATHER_API_KEY'))
    personalization_service = PersonalizationService()
    
    print("✅ Система готова!")
    yield
    print("👋 Завершение работы...")


app = FastAPI(
    title="Fashion AI API - Enhanced",
    description="Распознавание одежды с генерацией образов",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/")
async def root():
    return {"message": "Fashion AI API - Enhanced", "version": "2.0.0"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    user_id: Optional[str] = Form(None),
    style: str = Form("casual"),
    occasion: str = Form("casual"),
    season: str = Form("spring"),
    city: Optional[str] = Form(None)
):
    """Анализ изображения и генерация образов"""
    start_time = time.time()
    
    # Читаем изображение
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Неверный формат изображения")
    
    image_id = f"img_{abs(hash(contents)) % 10000000}"
    
    # 1. Детекция
    detections = fashion_detector.detect(image, return_crops=True)
    
    # 2. Анализ цветов
    detected_items = []
    clothing_items = []
    
    for i, det in enumerate(detections):
        if det.cropped_image is None:
            continue
        
        colors = color_analyzer.extract_dominant_colors(det.cropped_image)
        primary_color = colors[0] if colors else None
        
        color_name = primary_color.color_name if primary_color else "unknown"
        color_hex = primary_color.hex if primary_color else "#808080"
        
        item_response = ClothingItemResponse(
            id=f"item_{i}",
            category=det.category_type,
            subcategory=det.class_name,
            color=color_name,
            color_hex=color_hex,
            confidence=det.confidence,
            bbox=list(det.bbox)
        )
        detected_items.append(item_response)
        
        # Для генератора образов
        clothing_items.append(ClothingItem(
            id=f"item_{i}",
            category=det.category_type,
            subcategory=det.class_name,
            color=color_name
        ))
    
    # 3. Получаем погоду
    weather_info = None
    if city and weather_service:
        weather = await weather_service.get_weather(city)
        if weather:
            weather_info = {
                "city": weather.city,
                "temperature": weather.temperature,
                "condition": weather.condition.value,
                "description": weather.description
            }
    
    # 4. Генерация образов
    try:
        style_enum = OutfitStyle(style.lower())
    except ValueError:
        style_enum = OutfitStyle.CASUAL
    
    outfits_data = outfit_generator.generate(
        clothing_items, style=style_enum, occasion=occasion, season=season
    )
    
    outfits_response = []
    for outfit in outfits_data:
        outfit_items = []
        for item in outfit.items:
            # Находим соответствующий detected_item
            for di in detected_items:
                if di.id == item.id:
                    outfit_items.append(di)
                    break
        
        outfits_response.append(OutfitRecommendation(
            items=outfit_items,
            style=outfit.style.value,
            score=outfit.score,
            reasoning=outfit.reasoning
        ))
    
    # 5. Персонализация (если есть user_id)
    if user_id and personalization_service:
        outfits_response = personalization_service.get_personalized_recommendations(
            user_id, [o.dict() for o in outfits_response]
        )
        outfits_response = [OutfitRecommendation(**o) for o in outfits_response]
    
    processing_time = (time.time() - start_time) * 1000
    
    return AnalysisResponse(
        image_id=image_id,
        detected_items=detected_items,
        outfits=outfits_response,
        weather_info=weather_info,
        processing_time_ms=processing_time
    )


@app.post("/feedback")
async def submit_feedback(user_id: str, outfit_id: str, rating: int, comment: Optional[str] = None):
    """Обратная связь о рекомендованном образе"""
    if personalization_service:
        personalization_service.record_outfit_choice(
            user_id=user_id,
            outfit={"id": outfit_id},
            rating=rating
        )
    return {"status": "ok", "message": "Спасибо за обратную связь!"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "detector": fashion_detector is not None,
            "color_analyzer": color_analyzer is not None,
            "outfit_generator": outfit_generator is not None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
