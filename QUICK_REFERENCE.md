# Быстрая справка по улучшенной системе

## Что реализовано

### 1. Fashion Detector (`app/models/fashion_detector.py`)
```python
from app.models.fashion_detector import FashionDetector

detector = FashionDetector("yolov8n.pt")  # Или fine-tuned веса
detections = detector.detect(image)

for det in detections:
    print(f"{det.class_name} ({det.category_type}) - {det.confidence:.2f}")
```
**Классы**: 13 категорий одежды (shirt, trousers, dress, skirt, etc.)

### 2. Color Analyzer (`app/models/color_analyzer.py`)
```python
from app.models.color_analyzer import ColorAnalyzer

analyzer = ColorAnalyzer(n_colors=3)
colors = analyzer.extract_dominant_colors(crop_image)

# Цветовая гармония между двумя вещами
score = analyzer.get_color_harmony_score(colors1, colors2)
```

### 3. Outfit Generator (`app/services/outfit_generator.py`)
```python
from app.services.outfit_generator import RuleBasedGenerator, OutfitStyle

generator = RuleBasedGenerator()
outfits = generator.generate(
    items=clothing_items,
    style=OutfitStyle.CASUAL,
    occasion="date",
    season="spring"
)
```

### 4. Weather API (`app/services/weather_api.py`)
```python
from app.services.weather_api import WeatherService

weather = WeatherService(api_key="YOUR_KEY")
info = await weather.get_weather("Moscow")
recommendations = weather.get_clothing_recommendations(info)
```

### 5. Personalization (`app/services/personalization.py`)
```python
from app.services.personalization import PersonalizationService

service = PersonalizationService()
profile = service.get_or_create_profile("user123")

# Записать выбор пользователя
service.record_outfit_choice("user123", outfit, rating=5)

# Получить персонализированные рекомендации
personalized = service.get_personalized_recommendations("user123", outfits)
```

---

## Быстрый старт (3 команды)

```bash
# 1. Установить зависимости
pip install fastapi uvicorn ultralytics opencv-python numpy scikit-learn aiohttp

# 2. Настроить окружение (опционально)
copy .env.example .env
# Отредактировать .env, добавить WEATHER_API_KEY

# 3. Запустить
uvicorn api_enhanced:app --reload
```

Тест: `curl http://localhost:8000/`

---

## API Endpoints

| Endpoint | Method | Описание |
|----------|--------|----------|
| `/` | GET | Проверка работы |
| `/analyze` | POST | Анализ + генерация образов |
| `/feedback` | POST | Обратная связь |
| `/health` | GET | Статус сервисов |
| `/docs` | GET | Swagger UI |

### Пример запроса
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@photo.jpg" \
  -F "user_id=123" \
  -F "style=casual" \
  -F "city=Moscow"
```

---

## Распределение по сложности

| Фича | Сложность | Статус |
|------|-----------|--------|
| Fashion detection (13 классов) | Medium | ✅ |
| Color analysis (K-Means) | Low | ✅ |
| Color harmony rules | Low | ✅ |
| Outfit generator | Medium | ✅ |
| FashionCLIP embeddings | Low | ✅ |
| Weather integration | Low | ✅ |
| User profiles | Medium | ✅ |
| Collaborative filtering | Medium | ✅ |
| Дообучение YOLO | High | 📖 Гайд |
| LLM integration | High | 💡 Идея |

---

## Где взять готовые веса

### Fashion YOLO
1. **Обучить самому**: `docs/TRAINING_GUIDE.md`
2. **Hugging Face**: `valentinafeve/yolov8_fashion`
3. **DeepFashion2**: Скачать и сконвертировать

### FashionCLIP
- Hugging Face: `patrickjohncyh/fashion-clip`

---

## Пример ответа API

```json
{
  "image_id": "img_1234567",
  "detected_items": [
    {
      "id": "item_0",
      "category": "top",
      "subcategory": "short_sleeved_shirt",
      "color": "синий",
      "color_hex": "#1e3a8a",
      "confidence": 0.89,
      "bbox": [100, 200, 300, 400]
    }
  ],
  "outfits": [
    {
      "items": [...],
      "style": "casual",
      "score": 0.85,
      "reasoning": "Синий + Бежевый, цветовая гармония"
    }
  ],
  "weather_info": {
    "temperature": 15,
    "condition": "cloudy"
  },
  "processing_time_ms": 245
}
```

---

## Полезные ссылки

- **Документация**: `SETUP_AND_IMPROVEMENTS.md`
- **Архитектура**: `docs/ARCHITECTURE_UPDATED.md`
- **Обучение моделей**: `docs/TRAINING_GUIDE.md`
- **Swagger**: http://localhost:8000/docs (после запуска)
