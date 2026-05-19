"""
Fashion Clothing Detector - Fine-tuned YOLOv8 на DeepFashion2
Распознаёт 13 категорий одежды
"""
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from PIL import Image
import cv2


@dataclass
class FashionDetectionResult:
    """Результат детекции одежды"""
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    class_name: str  # Название категории одежды
    category_type: str  # top, bottom, outerwear, shoes, accessory
    cropped_image: Optional[np.ndarray] = None


class FashionDetector:
    """
    Детектор одежды на базе YOLOv8
    
    Классы (DeepFashion2):
    0: short_sleeved_shirt (футболка/рубашка)
    1: long_sleeved_shirt (лонгслив/рубашка)
    2: short_sleeved_outwear (куртка кор. рукав)
    3: long_sleeved_outwear (пальто/куртка)
    4: vest (жилет)
    5: sling (топ на бретельках)
    6: shorts (шорты)
    7: trousers (брюки)
    8: skirt (юбка)
    9: short_sleeved_dress (платье кор. рукав)
    10: long_sleeved_dress (платье длин. рукав)
    11: vest_dress (сарафан)
    12: sling_dress (платье на бретельках)
    """
    
    # Маппинг категорий
    CLASS_NAMES = {
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
    
    # Группировка по типам (для outfit generation)
    CATEGORY_GROUPS = {
        'top': [0, 1, 4, 5],  # верх
        'outerwear': [2, 3],  # верхняя одежда
        'bottom': [6, 7, 8],  # низ
        'dress': [9, 10, 11, 12],  # платья
        'full_body': [9, 10, 11, 12],  # платья (заменяют top+bottom)
    }
    
    def __init__(
        self,
        model_path: str = "yolov8n.pt",  # Временно используем стандартную, пока нет fashion весов
        conf_threshold: float = 0.4,
        device: str = "auto"
    ):
        """
        Инициализация детектора
        
        Args:
            model_path: Путь к весам модели
                - Для fashion: скачать с https://github.com/switchablenorms/DeepFashion2
                - Или использовать: 'yolov8n-cls.pt' для классификации после детекции
            conf_threshold: Порог уверенности
            device: Устройство ('cuda', 'cpu', 'auto')
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.conf_threshold = conf_threshold
        
        # Загрузка модели
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model.to(self.device)
            print(f"✓ Fashion Detector загружен: {model_path} на {self.device}")
        except ImportError:
            raise ImportError("Установите ultralytics: pip install ultralytics")
        
        # Если используем стандартную YOLO - будем классифицировать crop'ы отдельно
        self.use_fine_tuned = 'fashion' in model_path.lower() or 'deepfashion' in model_path.lower()
    
    def detect(
        self,
        image: np.ndarray,
        return_crops: bool = True
    ) -> List[FashionDetectionResult]:
        """
        Детекция одежды на изображении
        
        Args:
            image: BGR изображение (OpenCV формат)
            return_crops: Возвращать обрезанные изображения
            
        Returns:
            Список FashionDetectionResult
        """
        # Конвертация BGR → RGB для YOLO
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Inference
        results = self.model(image_rgb, conf=self.conf_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Координаты bbox
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                
                # Получаем имя класса
                if self.use_fine_tuned and class_id in self.CLASS_NAMES:
                    class_name = self.CLASS_NAMES[class_id]
                    category_type = self._get_category_type(class_id)
                else:
                    # Для стандартной YOLO - используем базовые классы
                    class_name = self.model.names.get(class_id, f"class_{class_id}")
                    category_type = self._map_coco_to_fashion(class_name)
                
                # Обрезка изображения
                cropped = None
                if return_crops:
                    cropped = image[y1:y2, x1:x2].copy()
                    # Проверка минимального размера
                    if cropped.shape[0] < 30 or cropped.shape[1] < 30:
                        cropped = None
                
                detection = FashionDetectionResult(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=class_name,
                    category_type=category_type,
                    cropped_image=cropped
                )
                detections.append(detection)
        
        # Сортировка по уверенности
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        return detections
    
    def _get_category_type(self, class_id: int) -> str:
        """Определить тип категории по ID"""
        if class_id in self.CATEGORY_GROUPS['top']:
            return 'top'
        elif class_id in self.CATEGORY_GROUPS['outerwear']:
            return 'outerwear'
        elif class_id in self.CATEGORY_GROUPS['bottom']:
            return 'bottom'
        elif class_id in self.CATEGORY_GROUPS['dress']:
            return 'dress'
        return 'unknown'
    
    def _map_coco_to_fashion(self, coco_class: str) -> str:
        """Маппинг COCO классов на fashion категории"""
        mappings = {
            'person': 'full_body',
            'backpack': 'accessory',
            'handbag': 'accessory',
            'suitcase': 'accessory',
            'tie': 'accessory',
        }
        return mappings.get(coco_class, 'unknown')
    
    def detect_person_and_clothing(
        self,
        image: np.ndarray
    ) -> Tuple[List[FashionDetectionResult], List[FashionDetectionResult]]:
        """
        Двухэтапный подход:
        1. Найти людей
        2. Классифицировать одежду на каждом человеке
        
        Returns:
            (people_detections, clothing_detections)
        """
        all_detections = self.detect(image)
        
        people = [d for d in all_detections if d.class_name == 'person']
        clothing = [d for d in all_detections if d.class_name != 'person']
        
        return people, clothing


# === РЕКОМЕНДАЦИИ ПО МОДЕЛЯМ ===

"""
ГОТОВЫЕ МОДЕЛИ ДЛЯ FASHION DETECTION:

1. **YOLOv8 Fine-tuned на DeepFashion2** (РЕКОМЕНДУЕТСЯ)
   Скачать: https://github.com/switchablenorms/DeepFashion2
   Веса: yolov8_deepfashion2.pt
   Классы: 13 категорий одежды
   
   Использование:
   detector = FashionDetector("yolov8_deepfashion2.pt")

2. **Fashion-Detector (Hugging Face)**
   Модель: 'valentinafeve/yolov8_fashion'
   
   Использование:
   from transformers import pipeline
   detector = pipeline("object-detection", model="valentinafeve/yolov8_fashion")

3. **Двухэтапный подход** (Если нет fashion YOLO):
   - Этап 1: YOLOv8 person detector
   - Этап 2: Классификатор на crop'ах (EfficientNet)
   
   См. файл: two_stage_approach.md

4. **Дообучение своей модели**:
   См. файл: training_guide.md
"""


if __name__ == "__main__":
    # Тест
    detector = FashionDetector("yolov8n.pt")
    
    # Тестовое изображение
    image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    
    detections = detector.detect(image)
    print(f"Найдено {len(detections)} объектов")
    for d in detections:
        print(f"  - {d.class_name} ({d.category_type}): {d.confidence:.2f}")
