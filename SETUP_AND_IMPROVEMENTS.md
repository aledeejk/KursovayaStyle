# Улучшенная система распознавания одежды - Инструкция

## Краткое описание подхода

Система решает проблему распознавания деталей одежды (рубашка/джинсы/платье) вместо просто "person":

1. **Fashion Detector**: YOLOv8 с fine-tuning на DeepFashion2 (13 категорий одежды)
2. **Color Analysis**: K-Means для извлечения доминирующих цветов
3. **Rule-based Outfit Generator**: Генерация образов с учётом цветовой гармонии
4. **Weather Integration**: OpenWeatherMap API для сезонных рекомендаций
5. **Personalization**: Профили пользователей + collaborative filtering

---

## Установка и запуск

### 1. Установка зависимостей

```bash
cd KursovayaStyle

# Создать виртуальное окружение
python -m venv venv

# Windows
venv\Scripts\activate

# Установить зависимости
pip install fastapi uvicorn ultralytics opencv-python numpy pillow
pip install scikit-learn aiohttp python-dotenv
pip install transformers torch  # Для FashionCLIP (опционально)
```

### 2. Настройка окружения

```bash
# Скопировать пример настроек
copy .env.example .env

# Отредактировать .env:
# WEATHER_API_KEY=your_key_here (получить на openweathermap.org)
# FASHION_MODEL=yolov8n.pt (или путь к fine-tuned весам)
```

### 3. Запуск сервера

```bash
# Вариант 1: Стандартный запуск
uvicorn api_enhanced:app --reload --port 8000

# Вариант 2: Production
uvicorn api_enhanced:app --host 0.0.0.0 --port 8000 --workers 4
```

### 4. Тестирование

```bash
# GET запрос - проверка работы
curl http://localhost:8000/

# POST запрос - анализ изображения
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@test_image.jpg" \
  -F "style=casual" \
  -F "occasion=date" \
  -F "city=Moscow"

# С user_id для персонализации
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@outfit.jpg" \
  -F "user_id=user123" \
  -F "style=business"
```

### 5. Документация API

Откройте в браузере:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## Структура проекта

```
KursovayaStyle/
├── api_enhanced.py              # FastAPI сервер (основной)
├── .env.example                 # Пример переменных окружения
├── app/
│   ├── models/
│   │   ├── fashion_detector.py  # YOLO fashion detection
│   │   ├── color_analyzer.py    # Анализ цветов (K-Means)
│   │   └── embeddings.py        # FashionCLIP эмбеддинги
│   └── services/
│       ├── outfit_generator.py  # Rule-based генератор
│       ├── weather_api.py       # OpenWeatherMap интеграция
│       └── personalization.py   # Персонализация + CF
├── docs/
│   ├── ARCHITECTURE_UPDATED.md  # Обновлённая архитектура
│   └── TRAINING_GUIDE.md        # Гайд по дообучению
└── data/                        # Профили пользователей
```

---

## Итоговый список улучшений

### Реализованные компоненты

| Улучшение | Сложность | Статус | Файл |
|-----------|-----------|--------|------|
| **Fashion Detector** (13 классов одежды) | Medium | ✅ Реализовано | `fashion_detector.py` |
| **Color Analysis** (K-Means + гармония) | Low | ✅ Реализовано | `color_analyzer.py` |
| **Outfit Generator** (rule-based) | Medium | ✅ Реализовано | `outfit_generator.py` |
| **FashionCLIP Embeddings** | Low | ✅ Есть | `embeddings.py` |
| **Weather API Integration** | Low | ✅ Реализовано | `weather_api.py` |
| **Personalization** (профили + CF) | Medium | ✅ Реализовано | `personalization.py` |
| **Enhanced API** | Low | ✅ Реализовано | `api_enhanced.py` |

### Описание улучшений

#### 1. Fashion Detection (Medium)
**Проблема**: Стандартная YOLO видит только "person"
**Решение**: 
- Класс `FashionDetector` с поддержкой DeepFashion2 весов
- 13 категорий: shirt, trousers, dress, skirt, etc.
- Fallback: двухэтапный подход (person → classify crops)

**Готовые веса**:
- Скачать: https://github.com/switchablenorms/DeepFashion2
- Или использовать `yolov8n.pt` (будет детектить person, потом классифицировать)

#### 2. Color Analysis (Low)
**Метод**: K-Means clustering на изображении одежды
**Выход**:
- RGB значения
- HEX коды
- Процент покрытия
- Название цвета на русском

**Цветовая гармония**:
- Комплементарные цвета (красный ↔ зелёный)
- Совместимость при генерации образов

#### 3. Outfit Generator (Medium)
**Тип**: Rule-based с fashion правилами

**Правила**:
- Цветовая гармония (комплементарные цвета)
- Шаблоны стилей (casual: t-shirt + jeans + sneakers)
- Погодные условия (добавление outerwear при холоде)

**Пример**:
```
Вход: синяя рубашка
Выход: синяя рубашка + бежевые брюки + коричневые лоферы
Причина: Синий + Бежевый (нейтральный), Business стиль
```

#### 4. Weather Integration (Low)
**API**: OpenWeatherMap (1000 calls/day бесплатно)

**Учёт в рекомендациях**:
- `temp < 10°C` → добавить куртку, свитер
- `rain` → дождевик, непромокаемая обувь
- `temp > 25°C` → лёгкая одежда, шорты

**Использование**:
```bash
curl -X POST "http://localhost:8000/analyze" \
  -F "file=@image.jpg" \
  -F "city=Moscow"  # Учтёт погоду в Москве
```

#### 5. Personalization (Medium)
**User Profile**:
- Любимые/нежелательные цвета
- Предпочитаемые стили
- История выборов
- Размеры

**Методы**:
- Content-based filtering (по предпочтениям)
- Collaborative filtering (похожие пользователи)
- Feedback loop (обучение на оценках 1-5)

**API**:
```bash
# Установить предпочтения
POST /set_preferences
{ "user_id": "123", "favorite_colors": ["blue", "black"] }

# Получить персонализированные рекомендации
POST /analyze
{ "file": "image.jpg", "user_id": "123" }

# Дать обратную связь
POST /feedback
{ "user_id": "123", "outfit_id": "xyz", "rating": 5 }
```

#### 6. FashionCLIP Embeddings (Low)
**Модель**: `patrickjohncyh/fashion-clip` (Hugging Face)

**Возможности**:
- Эмбеддинги 512-dim для каждой вещи
- Поиск похожих по косинусному сходству
- Text-to-image поиск

**Использование** (опционально):
```python
from app.models.embeddings import FashionEmbeddingModel

model = FashionEmbeddingModel()
embedding = model.encode_image(crop)  # (512,) vector
```

---

## Датасеты для дообучения

### DeepFashion2 (рекомендуется)
- **URL**: https://github.com/switchablenorms/DeepFashion2
- **Размер**: 491K изображений
- **Классы**: 13 категорий одежды
- **Лицензия**: Research only

### Для русскоязычных/специфических категорий

**Свой датасет**:
1. Собрать изображения (500+ на класс)
2. Разметить в CVAT или Roboflow
3. Обучить YOLOv8 (см. `docs/TRAINING_GUIDE.md`)

**Примеры специфичных категорий**:
- Варенки, валенки (зимняя обувь)
- Шубы, дубленки (зимняя верхняя одежда)
- Косоворотки, сарафаны (национальная)
- Пуховики (современная зима)

**Гайд по обучению**: см. `docs/TRAINING_GUIDE.md`

---

## Чеклист для запуска в production

- [ ] Установить все зависимости
- [ ] Получить WEATHER_API_KEY (openweathermap.org)
- [ ] Скачать/обучить fashion YOLO веса (опционально)
- [ ] Протестировать API локально
- [ ] Настроить CORS для production
- [ ] Добавить rate limiting (middleware)
- [ ] Настроить логирование
- [ ] (Опционально) Docker развёртывание

---

## Примеры использования

### 1. Базовый анализ
```python
import requests

with open('outfit.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/analyze',
        files={'file': f},
        data={'style': 'casual', 'occasion': 'weekend'}
    )

result = response.json()
print(f"Найдено {len(result['detected_items'])} вещей")
for outfit in result['outfits']:
    print(f"Образ (score: {outfit['score']}): {outfit['reasoning']}")
```

### 2. С учётом погоды
```python
response = requests.post(
    'http://localhost:8000/analyze',
    files={'file': open('image.jpg', 'rb')},
    data={'city': 'Moscow', 'season': 'winter'}
)
```

### 3. С персонализацией
```python
# Установить предпочтения
requests.post(
    'http://localhost:8000/set_preferences',
    json={
        'user_id': 'user123',
        'preferred_styles': ['business', 'casual'],
        'favorite_colors': ['blue', 'black', 'white'],
        'disliked_colors': ['yellow', 'orange']
    }
)

# Получить персонализированные рекомендации
response = requests.post(
    'http://localhost:8000/analyze',
    files={'file': open('image.jpg', 'rb')},
    data={'user_id': 'user123'}
)
```

---

## Контакты и поддержка

- Документация API: http://localhost:8000/docs
- Обучение моделей: `docs/TRAINING_GUIDE.md`
- Архитектура: `docs/ARCHITECTURE_UPDATED.md`
