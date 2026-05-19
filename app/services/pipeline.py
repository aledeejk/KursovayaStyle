"""
Complete Processing Pipeline

End-to-end pipeline from image upload to outfit generation
"""
import asyncio
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
from PIL import Image
import numpy as np
import cv2
from pathlib import Path
import hashlib
from datetime import datetime

from app.models.clothing_detector import ClothingDetector, DetectionResult
from app.models.classifier import ClothingClassifier, ClassificationResult
from app.models.embeddings import FashionEmbeddingModel
from app.services.outfit_generator import (
    ClothingItem, Outfit, HybridGenerator, OutfitStyle
)


@dataclass
class ProcessingResult:
    """Complete processing result"""
    image_id: str
    original_image: np.ndarray
    detections: List[Dict] = field(default_factory=list)
    items: List[ClothingItem] = field(default_factory=list)
    outfits: List[Outfit] = field(default_factory=list)
    processing_time_ms: float = 0.0
    metadata: Dict = field(default_factory=dict)


class FashionPipeline:
    """
    Complete fashion recognition and recommendation pipeline
    """
    
    def __init__(
        self,
        detector_model: str = "yolov8x.pt",
        classifier_model: str = "efficientnet_b4",
        embedding_model: str = "fashionclip",
        device: str = "auto",
        enable_caching: bool = True
    ):
        """
        Initialize pipeline with all components
        
        Args:
            detector_model: YOLO model path
            classifier_model: Classifier model name
            embedding_model: Embedding model name
            device: Device to use
            enable_caching: Enable result caching
        """
        self.device = device
        self.enable_caching = enable_caching
        
        # Initialize models
        print("🚀 Initializing Fashion AI Pipeline...")
        
        self.detector = ClothingDetector(
            model_path=detector_model,
            device=device
        )
        
        self.classifier = ClothingClassifier(
            model_name=classifier_model,
            device=device
        )
        
        self.embedding_model = FashionEmbeddingModel(
            model_name=embedding_model,
            device=device
        )
        
        self.outfit_generator = HybridGenerator(
            use_rules=True,
            use_ml=True,
            use_llm=False
        )
        
        # Cache
        self._cache = {}
        
        print("✅ Pipeline initialized successfully!")
    
    def _generate_image_id(self, image: np.ndarray) -> str:
        """Generate unique ID for image"""
        hash_obj = hashlib.md5(image.tobytes())
        return hash_obj.hexdigest()[:16]
    
    def _preprocess_image(
        self,
        image: np.ndarray,
        target_size: tuple = (640, 640)
    ) -> np.ndarray:
        """
        Preprocess image for processing
        
        Steps:
        1. Resize if too large
        2. Convert color space
        3. Normalize
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if BGR (OpenCV default)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Resize if too large
        h, w = image_rgb.shape[:2]
        max_dim = max(h, w)
        
        if max_dim > max(target_size):
            scale = max(target_size) / max_dim
            new_w, new_h = int(w * scale), int(h * scale)
            image_rgb = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return image_rgb
    
    async def process_image(
        self,
        image: np.ndarray,
        generate_outfits: bool = True,
        style_preference: str = "casual",
        occasion: str = "casual",
        user_preferences: Optional[Dict] = None
    ) -> ProcessingResult:
        """
        Process image through complete pipeline
        
        Pipeline:
        1. Preprocessing
        2. Detection
        3. Classification
        4. Embedding extraction
        5. Outfit generation (optional)
        
        Args:
            image: Input image (numpy array)
            generate_outfits: Whether to generate outfit recommendations
            style_preference: Preferred style
            occasion: Occasion context
            user_preferences: User personalization data
            
        Returns:
            ProcessingResult with all information
        """
        import time
        start_time = time.time()
        
        # Generate image ID
        image_id = self._generate_image_id(image)
        
        # Check cache
        if self.enable_caching and image_id in self._cache:
            return self._cache[image_id]
        
        # Step 1: Preprocessing
        processed_image = self._preprocess_image(image)
        
        # Step 2: Detection
        detections = self._detect_clothing(processed_image)
        
        # Step 3: Classification & Embedding
        items = await self._extract_items(processed_image, detections)
        
        # Step 4: Outfit Generation
        outfits = []
        if generate_outfits and len(items) >= 2:
            outfits = self._generate_outfits(
                items,
                style_preference,
                occasion,
                user_preferences
            )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        
        # Build result
        result = ProcessingResult(
            image_id=image_id,
            original_image=processed_image,
            detections=detections,
            items=items,
            outfits=outfits,
            processing_time_ms=processing_time,
            metadata={
                'timestamp': datetime.now().isoformat(),
                'model_versions': {
                    'detector': 'yolov8x',
                    'classifier': 'efficientnet_b4',
                    'embedding': 'fashionclip'
                }
            }
        )
        
        # Cache result
        if self.enable_caching:
            self._cache[image_id] = result
        
        return result
    
    def _detect_clothing(
        self,
        image: np.ndarray
    ) -> List[Dict]:
        """
        Detect clothing items in image
        
        Returns:
            List of detection results with bboxes
        """
        # Target fashion-related classes
        fashion_classes = [
            "person",  # Will detect people, then use crops
        ]
        
        detections = self.detector.detect(
            image,
            target_classes=None,  # Detect all
            return_crops=True
        )
        
        # Convert to dict format
        detection_results = []
        for det in detections:
            detection_results.append({
                'bbox': det.bbox,
                'confidence': det.confidence,
                'class_name': det.class_name,
                'class_id': det.class_id,
                'cropped_image': det.cropped_image
            })
        
        return detection_results
    
    async def _extract_items(
        self,
        image: np.ndarray,
        detections: List[Dict]
    ) -> List[ClothingItem]:
        """
        Extract clothing items with attributes
        
        For each detection:
        1. Classify category
        2. Extract attributes
        3. Generate embedding
        """
        items = []
        
        for i, det in enumerate(detections):
            cropped = det.get('cropped_image')
            if cropped is None or cropped.size == 0:
                continue
            
            # Ensure minimum size
            if cropped.shape[0] < 50 or cropped.shape[1] < 50:
                continue
            
            try:
                # Classification
                classification = self.classifier.classify(cropped)
                
                # Embedding
                embedding = self.embedding_model.encode_image(cropped)
                
                # Create item
                item = ClothingItem(
                    id=f"{det.get('class_name', 'item')}_{i}",
                    category=self._map_to_category(classification.category),
                    subcategory=classification.category,
                    color=classification.attributes.get('color', 'unknown'),
                    pattern=classification.attributes.get('pattern', 'solid'),
                    style=classification.attributes.get('style', 'casual'),
                    embedding=embedding,
                    attributes={
                        'confidence': classification.confidence,
                        'bbox': det['bbox'],
                        'detection_confidence': det['confidence'],
                        **classification.attributes
                    }
                )
                
                items.append(item)
                
            except Exception as e:
                print(f"Error processing detection {i}: {e}")
                continue
        
        return items
    
    def _map_to_category(self, subcategory: str) -> str:
        """Map subcategory to main category"""
        category_map = {
            't_shirt': 'top',
            'shirt': 'top',
            'sweater': 'top',
            'cardigan': 'top',
            'jacket': 'outerwear',
            'coat': 'outerwear',
            'dress': 'dress',
            'skirt': 'bottom',
            'pants': 'bottom',
            'shorts': 'bottom',
            'jeans': 'bottom',
            'leggings': 'bottom',
            'suit': 'outerwear',
            'blazer': 'outerwear',
            'vest': 'top',
        }
        return category_map.get(subcategory, 'other')
    
    def _generate_outfits(
        self,
        items: List[ClothingItem],
        style: str,
        occasion: str,
        user_preferences: Optional[Dict]
    ) -> List[Outfit]:
        """
        Generate outfit recommendations
        """
        try:
            style_enum = OutfitStyle(style.lower())
        except ValueError:
            style_enum = OutfitStyle.CASUAL
        
        outfits = self.outfit_generator.generate(
            items=items,
            style=style_enum,
            occasion=occasion,
            user_preferences=user_preferences,
            top_k=3
        )
        
        return outfits
    
    async def process_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 4
    ) -> List[ProcessingResult]:
        """
        Process multiple images in batch
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            
            # Process each image
            batch_tasks = [
                self.process_image(img)
                for img in batch
            ]
            
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
        
        return results
    
    def visualize_result(
        self,
        result: ProcessingResult,
        draw_embeddings: bool = False
    ) -> np.ndarray:
        """
        Visualize processing result
        
        Draws:
        - Bounding boxes
        - Labels
        - Outfit connections (if draw_embeddings=True)
        """
        image = result.original_image.copy()
        
        # Draw detections
        for item in result.items:
            if 'bbox' in item.attributes:
                x1, y1, x2, y2 = item.attributes['bbox']
                
                # Draw box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw label
                label = f"{item.subcategory} ({item.color})"
                cv2.putText(
                    image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
        
        return image


# Convenience function for quick processing
def process_wardrobe_image(
    image_path: str,
    style: str = "casual",
    occasion: str = "casual"
) -> ProcessingResult:
    """
    Quick process function for wardrobe image
    
    Example:
        result = process_wardrobe_image("my_outfit.jpg", "casual", "weekend")
    """
    # Initialize pipeline (singleton pattern in production)
    pipeline = FashionPipeline()
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Process
    result = asyncio.run(pipeline.process_image(
        image,
        generate_outfits=True,
        style_preference=style,
        occasion=occasion
    ))
    
    return result


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        # Initialize pipeline
        pipeline = FashionPipeline()
        
        # Load test image
        image = cv2.imread("test_wardrobe.jpg")
        
        if image is not None:
            # Process
            result = await pipeline.process_image(
                image,
                generate_outfits=True,
                style_preference="casual",
                occasion="date"
            )
            
            # Print results
            print(f"\n{'='*50}")
            print(f"Processing complete in {result.processing_time_ms:.0f}ms")
            print(f"Found {len(result.items)} clothing items:")
            
            for item in result.items:
                print(f"  - {item.color} {item.subcategory} ({item.style})")
            
            print(f"\nGenerated {len(result.outfits)} outfit suggestions:")
            for i, outfit in enumerate(result.outfits):
                print(f"\n  Outfit {i+1} (score: {outfit.score:.2f}):")
                for item in outfit.items:
                    print(f"    - {item.color} {item.subcategory}")
                print(f"    Reasoning: {outfit.reasoning}")
    
    # Run
    asyncio.run(main())
