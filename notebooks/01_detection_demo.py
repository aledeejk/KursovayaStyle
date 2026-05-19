"""
Demo: Clothing Detection with YOLOv8
"""
import cv2
import numpy as np
from pathlib import Path

# Добавляем путь к проекту
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.clothing_detector import ClothingDetector


def main():
    # Инициализация детектора
    print("Initializing YOLOv8 detector...")
    detector = ClothingDetector(
        model_path="yolov8x.pt",
        conf_threshold=0.5,
        device="auto"
    )
    
    # Загрузка тестового изображения
    test_image_path = "data/test_image.jpg"
    image = cv2.imread(test_image_path)
    
    if image is None:
        # Создаём тестовое изображение для демо
        print("Creating demo image...")
        image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        # Рисуем прямоугольник, имитирующий одежду
        cv2.rectangle(image, (100, 100), (300, 400), (100, 150, 200), -1)
    
    print(f"Image shape: {image.shape}")
    
    # Детекция
    print("\nRunning detection...")
    detections = detector.detect(image, return_crops=True)
    
    # Вывод результатов
    print(f"\nFound {len(detections)} items:")
    for i, det in enumerate(detections):
        print(f"  {i+1}. {det.class_name}")
        print(f"     Confidence: {det.confidence:.2f}")
        print(f"     Bbox: {det.bbox}")
        if det.cropped_image is not None:
            print(f"     Crop size: {det.cropped_image.shape}")
    
    # Визуализация
    result_image = detector.draw_detections(image, detections)
    
    # Сохранение результата
    output_path = "output_detection.jpg"
    cv2.imwrite(output_path, result_image)
    print(f"\nResult saved to: {output_path}")
    
    return detections


if __name__ == "__main__":
    main()
