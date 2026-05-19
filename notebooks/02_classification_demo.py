"""
Demo: Clothing Classification
"""
import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.classifier import SimpleAttributeClassifier


def main():
    # Инициализация
    print("Initializing classifier...")
    classifier = SimpleAttributeClassifier()
    
    # Создаём тестовое изображение
    # В реальности: загрузка crop одежды из детектора
    image = np.ones((300, 200, 3), dtype=np.uint8) * 150
    
    # Красный оттенок для теста
    image[:, :, 2] = 180
    
    print(f"\nImage shape: {image.shape}")
    
    # Извлечение цвета
    print("\nExtracting colors...")
    color_info = classifier.extract_color(image)
    
    print("Dominant colors:")
    for i, color in enumerate(color_info['dominant_colors']):
        rgb = color['rgb']
        print(f"  {i+1}. RGB{rgb} - {color['percentage']}%")
    
    # Классификация текстуры
    print("\nAnalyzing texture...")
    texture_scores = classifier.classify_texture(image)
    
    print("Texture scores:")
    for texture, score in sorted(texture_scores.items(), key=lambda x: -x[1]):
        bar = "█" * int(score * 20)
        print(f"  {texture:12s}: {score:.2f} {bar}")
    
    return color_info, texture_scores


if __name__ == "__main__":
    main()
