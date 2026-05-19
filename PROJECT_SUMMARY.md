# Fashion AI System - Проект распознавания одежды и генерации образов

## 📋 Сводка проекта

Этот проект представляет собой полноценную систему для распознавания одежды на изображениях и генерации стильных образов.

## 🎯 Основные возможности

1. **Распознавание одежды**: Детекция и классификация предметов одежды
2. **Извлечение атрибутов**: Цвет, узор, стиль, категория
3. **Векторные представления**: FashionCLIP embeddings для поиска
4. **Генерация образов**: Rule-based + ML подходы
5. **API**: REST API на FastAPI для интеграции

## 📁 Структура проекта

```
fashion_ai_system/
├── app/
│   ├── api/
│   │   └── main.py              # FastAPI приложение
│   ├── models/
│   │   ├── clothing_detector.py # YOLOv8 детектор
│   │   ├── classifier.py        # EfficientNet классификатор
│   │   └── embeddings.py        # FashionCLIP эмбеддинги
│   └── services/
│       ├── pipeline.py          # Полный pipeline
│       └── outfit_generator.py  # Генератор образов
├── docs/
│   ├── ARCHITECTURE.md          # Архитектура системы
│   ├── MODELS.md                # Выбор моделей
│   ├── DATASETS.md              # Рекомендуемые датасеты
│   ├── IMPROVEMENTS.md          # Идеи улучшений
│   └── DEPLOYMENT.md            # Развёртывание
├── notebooks/
│   ├── 01_detection_demo.py     # Демо детекции
│   ├── 02_classification_demo.py # Демо классификации
│   ├── 03_embeddings_demo.py    # Демо эмбеддингов
│   └── 04_outfit_generation_demo.py # Демо генерации
├── tests/
│   ├── test_detector.py         # Тесты детектора
│   └── test_outfit_generator.py # Тесты генератора
├── examples/
│   └── quick_start.py           # Быстрый старт
├── requirements.txt             # Зависимости
├── Dockerfile                   # Docker образ
├── docker-compose.yml           # Docker Compose
└── README.md                    # Основная документация
```

## 🚀 Быстрый старт

```bash
# 1. Клонирование и установка
cd fashion_ai_system
pip install -r requirements.txt

# 2. Запуск демо
python examples/quick_start.py

# 3. Запуск API
uvicorn app.api.main:app --reload

# 4. Docker развёртывание
docker-compose up -d
```

## 🔧 Архитектура системы

### Компоненты

| Компонент | Модель | Назначение |
|-----------|--------|------------|
| Detection | YOLOv8-x | Обнаружение предметов одежды |
| Classification | EfficientNet-B4 | Классификация категорий |
| Embeddings | FashionCLIP | Векторные представления |
| Outfit Gen | Hybrid | Rule-based + ML |

### Pipeline обработки

```
Image Upload
    ↓
Preprocessing
    ↓
Detection (YOLOv8) → bboxes
    ↓
Classification → category, color, pattern, style
    ↓
Embedding (CLIP) → 512-dim vector
    ↓
Outfit Generation
    ├─ Rule-based: color harmony, fashion rules
    ├─ ML-based: embedding similarity
    └─ Scoring: outfit compatibility
    ↓
Results: detected items + outfit recommendations
```

## 📊 Технологический стек

### Computer Vision
- PyTorch, TorchVision
- Ultralytics (YOLO)
- transformers (CLIP)
- timm (EfficientNet)
- OpenCV

### Backend
- FastAPI (async)
- Pydantic v2
- SQLAlchemy
- PostgreSQL + pgvector
- Redis

### Infrastructure
- Docker
- Docker Compose
- Nginx

## 📈 Производительность

| Операция | Время (GPU RTX 3090) |
|----------|---------------------|
| Detection | ~30ms |
| Classification | ~15ms |
| Embedding | ~50ms |
| Full Pipeline | ~150-200ms |

## 🎓 Модели

### Pre-trained (готовы к использованию)
1. **YOLOv8**: Детекция объектов
2. **EfficientNet-B4**: Классификация
3. **FashionCLIP**: Эмбеддинги (patrickjohncyh/fashion-clip)

### Fine-tuning (опционально)
- DeepFashion2 для fashion-specific detection
- Поликовер для outfit compatibility

## 📚 Датасеты

### Для обучения
- **DeepFashion2**: 491K изображений, 13 категорий
- **Fashionpedia**: 48K изображений, 27 категорий
- **Polyvore**: 68K outfits, outfit compatibility

### Для тестирования
- Собственные изображения
- Fashion-MNIST (базовые эксперименты)

## 💡 Идеи для улучшения

1. **Персонализация**: User profiles, feedback loop
2. **Погода**: Weather API integration
3. **LLM**: GPT-4 для текстовых рекомендаций
4. **Социальные**: Community, sharing, trending outfits
5. **Мобильное**: On-device ML, camera integration
6. **Virtual Try-on**: 3D fitting, AR

## 🔐 Безопасность

- Rate limiting
- API key authentication
- Image size validation
- Encrypted storage
- GDPR compliance

## 📝 Лицензия

MIT License - открытый исходный код

## 🤝 Контрибуция

1. Fork репозитория
2. Создайте feature branch
3. Commit изменения
4. Push и создайте Pull Request

## 📞 Контакты

- GitHub: [repository-url]
- Email: [contact-email]
- Docs: [documentation-url]

---

**Статус проекта**: Production-ready MVP
**Версия**: 1.0.0
**Последнее обновление**: Май 2026
