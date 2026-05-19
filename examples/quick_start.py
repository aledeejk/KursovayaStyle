"""
Quick Start Example - Полный пример работы системы
"""
import asyncio
import numpy as np
import cv2

# Импорты из проекта
from app.models.clothing_detector import ClothingDetector
from app.models.classifier import SimpleAttributeClassifier
from app.models.embeddings import FashionEmbeddingModel
from app.services.outfit_generator import ClothingItem, RuleBasedGenerator, OutfitStyle
from app.services.pipeline import FashionPipeline


def demo_detection():
    """Demo: Детекция одежды"""
    print("="*60)
    print("DEMO 1: Clothing Detection")
    print("="*60)
    
    # Инициализация
    detector = ClothingDetector(model_path="yolov8n.pt")
    
    # Тестовое изображение
    image = np.ones((640, 480, 3), dtype=np.uint8) * 220
    cv2.rectangle(image, (100, 100), (300, 500), (100, 150, 200), -1)
    
    # Детекция
    detections = detector.detect(image)
    
    print(f"\nНайдено {len(detections)} объектов:")
    for det in detections:
        print(f"  - {det.class_name}: {det.confidence:.2f}")
    
    return detections


def demo_classification():
    """Demo: Классификация атрибутов"""
    print("\n" + "="*60)
    print("DEMO 2: Attribute Classification")
    print("="*60)
    
    classifier = SimpleAttributeClassifier()
    
    # Создаём тестовое изображение одежды
    clothing_img = np.ones((300, 200, 3), dtype=np.uint8)
    clothing_img[:, :] = [50, 50, 180]  # Синий цвет
    
    # Извлечение цвета
    color_info = classifier.extract_color(clothing_img)
    print(f"\nДоминирующие цвета:")
    for i, c in enumerate(color_info['dominant_colors'][:3]):
        print(f"  {i+1}. RGB{c['rgb']} - {c['percentage']}%")
    
    # Классификация текстуры
    texture = classifier.classify_texture(clothing_img)
    print(f"\nТекстура: {max(texture, key=texture.get)}")
    
    return color_info


def demo_embeddings():
    """Demo: Fashion Embeddings"""
    print("\n" + "="*60)
    print("DEMO 3: Fashion Embeddings (CLIP)")
    print("="*60)
    
    # Инициализация (загрузка модели)
    print("\nЗагрузка FashionCLIP...")
    # model = FashionEmbeddingModel(model_name='fashionclip')
    print("(Пропущено для демо - требуется загрузка модели)")
    
    print("\nВозможности:")
    print("  - Извлечение 512-мерных эмбеддингов из изображений")
    print("  - Сравнение похожести образов")
    print("  - Текстовый поиск по описанию")
    print("  - Zero-shot классификация")


def demo_outfit_generation():
    """Demo: Генерация образов"""
    print("\n" + "="*60)
    print("DEMO 4: Outfit Generation")
    print("="*60)
    
    # Тестовый гардероб
    wardrobe = [
        ClothingItem("1", "top", "t_shirt", "white", "solid", "casual"),
        ClothingItem("2", "top", "shirt", "blue", "striped", "business"),
        ClothingItem("3", "bottom", "jeans", "blue", "solid", "casual"),
        ClothingItem("4", "bottom", "pants", "black", "solid", "business"),
        ClothingItem("5", "shoes", "sneakers", "white", "solid", "casual"),
        ClothingItem("6", "shoes", "loafers", "brown", "solid", "business"),
    ]
    
    generator = RuleBasedGenerator()
    
    # Генерация casual образа
    print("\nCasual outfit для выходных:")
    outfits = generator.generate(
        wardrobe, 
        style=OutfitStyle.CASUAL,
        occasion="weekend",
        season="summer"
    )
    
    if outfits:
        outfit = outfits[0]
        print(f"  Score: {outfit.score:.2f}")
        for item in outfit.items:
            print(f"  - {item.color} {item.subcategory}")
        print(f"  Обоснование: {outfit.reasoning}")
    
    # Генерация business образа
    print("\nBusiness outfit для работы:")
    outfits = generator.generate(
        wardrobe,
        style=OutfitStyle.BUSINESS,
        occasion="work",
        season="spring"
    )
    
    if outfits:
        outfit = outfits[0]
        print(f"  Score: {outfit.score:.2f}")
        for item in outfit.items:
            print(f"  - {item.color} {item.subcategory}")
        print(f"  Обоснование: {outfit.reasoning}")


def demo_full_pipeline():
    """Demo: Полный pipeline"""
    print("\n" + "="*60)
    print("DEMO 5: Full Pipeline (Сводка)")
    print("="*60)
    
    print("""
Полный pipeline обработки:

1. Загрузка изображения
   ↓
2. Предобработка (resize, normalize)
   ↓
3. Детекция (YOLOv8) → bounding boxes
   ↓
4. Классификация каждого item
   ├─ Category: dress, shirt, pants...
   ├─ Color: red, blue, black...
   ├─ Pattern: solid, striped...
   └─ Style: casual, formal...
   ↓
5. Экстракция embeddings (FashionCLIP)
   ↓
6. Сохранение в векторную БД
   ↓
7. Генерация outfit рекомендаций
   ├─ Rule-based подбор
   ├─ ML similarity matching
   └─ Совместимость scoring
   ↓
8. Возврат результатов пользователю

Время обработки: ~150-200ms на GPU
    """)


def main():
    """Запуск всех демо"""
    print("\n" + "="*60)
    print("FASHION AI SYSTEM - QUICK START DEMO")
    print("="*60)
    
    try:
        demo_detection()
    except Exception as e:
        print(f"Detection demo skipped: {e}")
    
    try:
        demo_classification()
    except Exception as e:
        print(f"Classification demo skipped: {e}")
    
    demo_embeddings()
    
    try:
        demo_outfit_generation()
    except Exception as e:
        print(f"Outfit generation demo skipped: {e}")
    
    demo_full_pipeline()
    
    print("\n" + "="*60)
    print("Демо завершено!")
    print("="*60)
    print("""
Следующие шаги:
1. Установите зависимости: pip install -r requirements.txt
2. Скачайте модели (запустите notebooks)
3. Запустите API: uvicorn app.api.main:app --reload
4. Откройте документацию: http://localhost:8000/docs
    """)


if __name__ == "__main__":
    main()
