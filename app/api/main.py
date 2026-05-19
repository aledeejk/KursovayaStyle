"""
FastAPI Application for Fashion AI System
"""
import io
import base64
from typing import List, Optional
from contextlib import asynccontextmanager

import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.services.pipeline import FashionPipeline


class ClothingItemResponse(BaseModel):
    id: str
    category: str
    subcategory: str
    color: str
    pattern: str
    style: str
    attributes: dict


class OutfitResponse(BaseModel):
    items: List[ClothingItemResponse]
    style: str
    occasion: str
    score: float
    reasoning: str


class ProcessingResponse(BaseModel):
    image_id: str
    items: List[ClothingItemResponse]
    outfits: List[OutfitResponse]
    processing_time_ms: float


pipeline: Optional[FashionPipeline] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("Starting Fashion AI System...")
    pipeline = FashionPipeline(device="auto")
    print("Ready!")
    yield
    print("Shutting down...")


app = FastAPI(
    title="Fashion AI API",
    description="Clothing recognition and outfit generation",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Fashion AI API", "version": "1.0.0"}


@app.post("/analyze", response_model=ProcessingResponse)
async def analyze_image(
    file: UploadFile = File(...),
    style: str = "casual",
    occasion: str = "casual"
):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    
    # Read image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image")
    
    # Process
    result = await pipeline.process_image(
        image,
        generate_outfits=True,
        style_preference=style,
        occasion=occasion
    )
    
    # Build response
    items = [
        ClothingItemResponse(
            id=item.id,
            category=item.category,
            subcategory=item.subcategory,
            color=item.color,
            pattern=item.pattern,
            style=item.style,
            attributes=item.attributes
        )
        for item in result.items
    ]
    
    outfits = [
        OutfitResponse(
            items=[
                ClothingItemResponse(
                    id=i.id, category=i.category, subcategory=i.subcategory,
                    color=i.color, pattern=i.pattern, style=i.style,
                    attributes=i.attributes
                ) for i in outfit.items
            ],
            style=outfit.style.value,
            occasion=outfit.occasion,
            score=outfit.score,
            reasoning=outfit.reasoning
        )
        for outfit in result.outfits
    ]
    
    return ProcessingResponse(
        image_id=result.image_id,
        items=items,
        outfits=outfits,
        processing_time_ms=result.processing_time_ms
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "pipeline_ready": pipeline is not None}
