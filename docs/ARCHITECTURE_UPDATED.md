# Обновлённая архитектура Fashion AI System

## 1. Общая структура системы

```
┌────────────────────────────────────────────────────────────────┐
│                    FASHION AI SYSTEM v2.0                    │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │   Frontend   │────▶│  FastAPI     │────▶│   Services   │   │
│  │  (React/Web) │◀────│   Backend    │◀────│   (Async)    │   │
│  └──────────────┘     └──────────────┘     └──────┬───────┘   │
│                                                    │          │
│  ┌──────────────────────────────────────────────────┼──────┐│
│  │                     ML LAYER                       │      ││
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐    │      ││
│  │  │ Fashion    │ │ Color      │ │ Embeddings │◀───┘      ││
│  │  │ Detector   │ │ Analyzer   │ │ (CLIP)     │            ││
│  │  │ (YOLOv8)   │ │ (K-Means)  │ │ 512-dim    │            ││
│  │  └────────────┘ └────────────┘ └────────────┘            ││
│  └──────────────────────────────────────────────────────────┘│
│                                                              │
│  ┌──────────────────────────────────────────────────────────┐│
│  │              INTELLIGENT SERVICES                         ││
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐           ││
│  │  │ Outfit     │ │ Weather     │ │ Personal-  │           ││
│  │  │ Generator  │ │ Service     │ │ ization    │           ││
│  │  │ (Rules+ML) │ │ (OpenWeather)│ │ (CF+Prefs) │           ││
│  │  └────────────┘ └────────────┘ └────────────┘           ││
│  └──────────────────────────────────────────────────────────┘│
└────────────────────────────────────────────────────────────────┘
```

## 2. Поток данных (Pipeline)

```
Загрузка изображения
      ↓
┌─────────────────┐
│ 1. Detection    │ YOLOv8 Fashion (13 классов)
│    - person     │
│    - clothing   │ BBox + category
└────────┬────────┘
         ↓
┌─────────────────┐
│ 2. Color        │ K-Means clustering
│    Analysis     │ 3-5 доминирующих цветов
│                 │ RGB → Name (RU)
└────────┬────────┘
         ↓
┌─────────────────┐
│ 3. Outfit       │ Rule-based генерация
│    Generation   │ • Color harmony (комплементарные)
│                 │ • Style templates
│                 │ • Weather rules
└────────┬────────┘
         ↓
┌─────────────────┐
│ 4. Personalize  │ User preferences
│                 │ • Color filtering
│                 │ • Style history
│                 │ • Collaborative filtering
└────────┬────────┘
         ↓
    Результат: JSON с образами
```

## 3. Компоненты системы

### 3.1 Fashion Detection (`app/models/fashion_detector.py`)

**Модель**: YOLOv8 fine-tuned на DeepFashion2

**Классы одежды** (13 категорий):
- **Верх**: short_sleeved_shirt, long_sleeved_shirt, vest, sling
- **Низ**: shorts, trousers, skirt  
- **Верхняя одежда**: short_sleeved_outwear, long_sleeved_outwear
- **Платья**: short_sleeved_dress, long_sleeved_dress, vest_dress, sling_dress

**Альтернативы**:
- Двухэтапный: YOLO person detection + EfficientNet classifier на crop'ах
- Hugging Face: `valentinafeve/yolov8_fashion`

### 3.2 Color Analysis (`app/models/color_analyzer.py`)

**Метод**: K-Means clustering (scikit-learn)

**Вход**: Crop одежды (numpy array)
**Выход**: 
- RGB значения
- HEX коды
- Процент покрытия
- Название цвета (русское)

**Цветовая гармония**:
- Комплементарные цвета (противоположные на круге)
- Аналогичные (рядом на круге)
- Нейтральные (black, white, gray, beige)

### 3.3 Outfit Generation (`app/services/outfit_generator.py`)

**Rule-based подход**:

```python
Правила:
1. Цветовая гармония:
   - Красный ↔ Зелёный (комплементарные)
   - Синий ↔ Оранжевый
   - Жёлтый ↔ Фиолетовый

2. Типы образов:
   casual: [t_shirt, jeans, sneakers]
   formal: [dress_shirt, suit_pants, dress_shoes, blazer]
   business: [dress_shirt, dress_pants, loafers]

3. Погодные правила:
   temp < 10°C → добавить outerwear
   rain → добавить дождевик
   snow → зимняя обувь
```

### 3.4 Weather Integration (`app/services/weather_api.py`)

**API**: OpenWeatherMap (бесплатный: 1000 calls/day)

**Параметры**:
- Температура (°C)
- Осадки (rain/snow)
- Влажность (%)
- Ветер (м/с)

**Рекомендации**:
- Cold (<10°C): куртка, свитер, шарф
- Rain: дождевик, непромокаемая обувь
- Hot (>25°C): лёгкая одежда, головной убор

### 3.5 Personalization (`app/services/personalization.py`)

**User Profile**:
```json
{
  "user_id": "uuid",
  "preferred_styles": ["casual", "business"],
  "favorite_colors": ["blue", "black"],
  "disliked_colors": ["yellow"],
  "style_history": [...],
  "size_info": {"top": "M", "bottom": "L"}
}
```

**Методы**:
- Content-based: по предпочтениям пользователя
- Collaborative filtering: похожие пользователи
- Feedback loop: обучение на оценках

## 4. API Endpoints

### Основные

```python
POST /analyze
  └─ Анализ изображения + генерация образов
  Params: file, user_id, style, occasion, season, city
  
POST /feedback  
  └─ Обратная связь о рекомендации
  Params: user_id, outfit_id, rating

GET /health
  └─ Статус сервисов
```

### Пример запроса

```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@outfit.jpg" \
  -F "user_id=user123" \
  -F "style=casual" \
  -F "occasion=date" \
  -F "city=Moscow"
```

### Пример ответа

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
    },
    {
      "id": "item_1", 
      "category": "bottom",
      "subcategory": "jeans",
      "color": "синий",
      "confidence": 0.92,
      "bbox": [100, 400, 300, 600]
    }
  ],
  "outfits": [
    {
      "items": [...],
      "style": "casual",
      "score": 0.85,
      "reasoning": "Синий + Синий (монохром), casual стиль"
    }
  ],
  "weather_info": {
    "city": "Moscow",
    "temperature": 15,
    "condition": "cloudy"
  },
  "processing_time_ms": 245.3
}
```

## 5. Модели данных

### ClothingItem
```python
@dataclass
class ClothingItem:
    id: str                    # UUID
    category: str              # top/bottom/outerwear/dress
    subcategory: str           # конкретный тип
    color: str                 # название цвета (RU)
    pattern: str = "solid"     # solid/striped/checkered
    style: str = "casual"      # casual/formal/business
    embedding: Optional[np.ndarray] = None  # 512-dim
```

### Outfit
```python
@dataclass
class Outfit:
    items: List[ClothingItem]
    style: OutfitStyle
    occasion: str
    weather: Optional[WeatherCondition]
    score: float              # compatibility score 0-1
    reasoning: str            # объяснение рекомендации
```

## 6. Технологический стек

| Компонент | Технология |
|-----------|-----------|
| Detection | YOLOv8 (Ultralytics) + DeepFashion2 |
| Color Analysis | OpenCV + scikit-learn (K-Means) |
| Embeddings | FashionCLIP (Hugging Face) |
| Backend | FastAPI + Pydantic |
| Weather | OpenWeatherMap API + aiohttp |
| Personalization | JSON storage + numpy |
| Async | asyncio + aiohttp |

## 7. Переменные окружения

```bash
# Модели
FASHION_MODEL=yolov8n.pt  # или путь к fine-tuned весам

# Weather API
WEATHER_API_KEY=your_openweathermap_key

# Storage
DATA_DIR=./data
```

## 8. Производительность

| Операция | Время (CPU) | Время (GPU) |
|----------|-------------|-------------|
| Detection | 50-100ms | 20-30ms |
| Color Analysis | 30-50ms | 30-50ms |
| Outfit Gen | 5-10ms | 5-10ms |
| **Total** | **100-200ms** | **60-100ms** |
