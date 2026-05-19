"""
Clothing Detection Module using YOLOv8
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import cv2


@dataclass
class DetectionResult:
    """Result of clothing detection"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str
    cropped_image: Optional[np.ndarray] = None


class ClothingDetector:
    """
    Clothing detector based on YOLOv8
    
    Supported classes (COCO + DeepFashion):
    - person, dress, shirt, pants, skirt, jacket, shoes, etc.
    """
    
    # Fashion-related COCO classes
    FASHION_CLASSES = {
        0: 'person',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        31: 'handbag',
        32: 'tie',
        33: 'suitcase',
    }
    
    # DeepFashion2 classes (if fine-tuned)
    DEEPFASHION_CLASSES = {
        0: 'short_sleeved_shirt',
        1: 'long_sleeved_shirt',
        2: 'short_sleeved_outwear',
        3: 'long_sleeved_outwear',
        4: 'vest',
        5: 'sling',
        6: 'shorts',
        7: 'trousers',
        8: 'skirt',
        9: 'short_sleeved_dress',
        10: 'long_sleeved_dress',
        11: 'vest_dress',
        12: 'sling_dress',
    }
    
    def __init__(
        self,
        model_path: str = "yolov8x.pt",
        conf_threshold: float = 0.5,
        device: str = "auto"
    ):
        """
        Initialize detector
        
        Args:
            model_path: Path to YOLO model or model name
            conf_threshold: Confidence threshold for detections
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.conf_threshold = conf_threshold
        
        # Setup device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"✓ Loaded YOLO model: {model_path} on {self.device}")
        except ImportError:
            raise ImportError("Install ultralytics: pip install ultralytics")
    
    def detect(
        self,
        image: np.ndarray,
        return_crops: bool = True,
        target_classes: Optional[List[str]] = None
    ) -> List[DetectionResult]:
        """
        Detect clothing items in image
        
        Args:
            image: Input image (BGR format from OpenCV or RGB)
            return_crops: Whether to return cropped images
            target_classes: Filter by specific classes (None = all)
            
        Returns:
            List of DetectionResult objects
        """
        # Run inference
        results = self.model(image, conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Get class name
                class_name = self.model.names.get(class_id, f"class_{class_id}")
                
                # Filter by target classes if specified
                if target_classes and class_name not in target_classes:
                    continue
                
                # Extract cropped image
                cropped = None
                if return_crops:
                    cropped = image[y1:y2, x1:x2].copy()
                
                detection = DetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    cropped_image=cropped
                )
                detections.append(detection)
        
        # Sort by confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections
    
    def detect_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 4
    ) -> List[List[DetectionResult]]:
        """Batch detection for multiple images"""
        all_detections = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            results = self.model(batch, conf=self.conf_threshold, verbose=False)
            
            for result in results:
                detections = []
                boxes = result.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                    
                    detection = DetectionResult(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=class_name
                    )
                    detections.append(detection)
                
                all_detections.append(detections)
        
        return all_detections
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[DetectionResult],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """Draw bounding boxes on image"""
        img_copy = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Draw box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{det.class_name}: {det.confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = y1 - 10 if y1 - 10 > 10 else y1 + 20
            
            cv2.rectangle(
                img_copy,
                (x1, label_y - label_size[1] - 5),
                (x1 + label_size[0], label_y + 5),
                color,
                -1
            )
            cv2.putText(
                img_copy, label, (x1, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            )
        
        return img_copy


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = ClothingDetector(
        model_path="yolov8x.pt",
        conf_threshold=0.5
    )
    
    # Load test image
    image = cv2.imread("test_image.jpg")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect clothing
    detections = detector.detect(image_rgb, target_classes=["person"])
    
    print(f"Found {len(detections)} items:")
    for det in detections:
        print(f"  - {det.class_name}: {det.confidence:.2f} at {det.bbox}")
    
    # Draw results
    result_image = detector.draw_detections(image, detections)
    cv2.imwrite("output.jpg", result_image)
