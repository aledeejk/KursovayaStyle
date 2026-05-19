# Архитектура Fashion AI System

## 1. Общая архитектура

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FASHION AI SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐     ┌──────────────┐     ┌─────────────────────┐      │
│  │   Frontend   │────▶│   FastAPI    │────▶│  ML Pipeline        │      │
│  │  (React/Web) │◀────│   Backend    │◀────│  (async)            │      │
│  └──────────────┘     └──────────────┘     └─────────────────────┘      │
│                                                      │                  │
│                              ┌───────────────────────┼─────────────┐    │
│                              ▼                       ▼             ▼    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    Модели (PyTorch/Transformers)                   │  │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────┐ │  │
│  │  │ Detection    │ │ Classification│ │  Embeddings  │ │ Outfit   │ │  │
│  │  │ (YOLOv8)     │ │ (EfficientNet) │ │  (CLIP)      │ │ Generator│ │  │
│  │  └──────────────┘ └──────────────┘ └──────────────┘ └──────────┘ │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Компоненты системы

### 2.1 Computer Vision Pipeline

| Этап | Модель | Назначение |
|------|--------|-----------|
| Detection | YOLOv8-x | Обнаружение предметов одежды |
| Classification | EfficientNet-B4 | Определение категории |
| Attributes | Custom CNN | Цвет, узор, стиль |
| Embedding | FashionCLIP | Векторное представление |

### 2.2 Outfit Generation

| Подход | Метод | Применение |
|--------|-------|-----------|
| Rule-based | Fashion rules + color theory | Быстрая генерация |
| ML-based | Embedding similarity | Персонализация |
| LLM-based | GPT-4/Claude | Сложные рекомендации |

## 3. Поток данных

```
Загрузка изображения
      ↓
Предобработка (resize, normalize)
      ↓
Detection (YOLOv8) → bbox + class
      ↓
Классификация (EfficientNet)
      ├─ Category: dress, shirt, pants...
      ├─ Color: red, blue, black...
      ├─ Pattern: solid, striped, floral...
      └─ Style: casual, formal, sporty...
      ↓
Embedding (FashionCLIP) → 512-dim vector
      ↓
Сохранение в Vector DB
      ↓
Outfit Generation
      ├─ Rule-based подбор
      ├─ ML similarity search
      └─ LLM рекомендации
      ↓
Возврат результатов
```

## 4. Технологический стек

### Backend
- **Framework**: FastAPI (async, high performance)
- **Validation**: Pydantic v2
- **ML**: PyTorch, Transformers, Ultralytics
- **Database**: PostgreSQL + pgvector
- **Cache**: Redis
- **Storage**: MinIO / S3

### ML Stack
- **Detection**: YOLOv8 (Ultralytics)
- **Classification**: timm (EfficientNet)
- **Embeddings**: Hugging Face Transformers (CLIP)
- **CV**: OpenCV, Pillow

### Infrastructure
- **Container**: Docker
- **Orchestration**: Kubernetes (optional)
- **API Gateway**: Nginx / Kong

## 5. Модели данных

### ClothingItem
```python
{
    id: str
    category: str           # top, bottom, shoes, accessory
    subcategory: str        # t-shirt, jeans, sneakers
    color: str
    pattern: str
    style: str
    embedding: np.ndarray   # 512-dim vector
    attributes: dict        # meta information
}
```

### Outfit
```python
{
    items: List[ClothingItem]
    style: OutfitStyle
    occasion: str
    weather: Optional[WeatherCondition]
    score: float           # compatibility score
    reasoning: str         # explanation
}
```
