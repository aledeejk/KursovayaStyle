# Fashion AI System: Clothing Recognition & Outfit Generation

Полноценная система для распознавания одежды на изображениях и генерации стильных образов.

## 🏗️ Архитектура системы

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FASHION AI SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐               │
│  │   Frontend   │──────▶   API GW     │──────▶   Backend    │               │
│  │  (React/Web) │◀─────│   (Nginx)    │◀─────│   (FastAPI)  │               │
│  └──────────────┘      └──────────────┘      └──────┬───────┘               │
│                                                     │                       │
│                              ┌──────────────────────┼───────────────────┐   │
│                              ▼                      ▼                   ▼   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    ML Services (Docker/K8s)                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │   Detection │  │ Classification│ │  Embedding  │  │  Outfit    │  │   │
│  │  │  (YOLOv8)   │  │ (EfficientNet)│ │  (CLIP)     │  │ Generator  │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                              │                      │                       │
│                              ▼                      ▼                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Data Layer                                 │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │   │
│  │  │ PostgreSQL  │  │    Redis    │  │    S3/MinIO │  │Elasticsearch│  │   │
│  │  │  (metadata) │  │   (cache)   │  │  (images)   │  │  (search)   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🔧 Стек технологий

### Computer Vision
| Задача | Модель | Библиотека |
|--------|--------|------------|
| Detection | YOLOv8-x | Ultralytics |
| Classification | EfficientNet-B4 | timm |
| Embeddings | CLIP / FashionCLIP | transformers |
| Segmentation | SAM | segment-anything |

### Backend
| Компонент | Технология |
|-----------|------------|
| Framework | FastAPI |
| ORM | SQLAlchemy 2.0 |
| Validation | Pydantic v2 |
| Async | asyncio, aiohttp |
| Task Queue | Celery + Redis |

### Хранение данных
| Тип | Технология |
|-----|------------|
| Metadata | PostgreSQL |
| Cache | Redis |
| Images | MinIO / S3 |
| Vector Search | pgvector / Elasticsearch |
| Logs | ClickHouse |

## 📊 Pipeline обработки

```
1. Upload Image
      ↓
2. Preprocessing (resize, normalize)
      ↓
3. Detection (YOLOv8) → bbox + class
      ↓
4. Classification (EfficientNet) → category
      ↓
5. Embedding (CLIP) → 512-dim vector
      ↓
6. Save to Vector DB
      ↓
7. Outfit Generation
      ↓
8. Return results
```

## 📁 Структура проекта

```
fashion_ai_system/
├── app/
│   ├── api/              # API endpoints
│   ├── core/             # Config, security
│   ├── models/           # ML models
│   ├── services/         # Business logic
│   └── db/               # Database models
├── ml_services/          # Microservices for ML
├── notebooks/            # Research & experiments
├── tests/                # Test suite
└── docker/               # Docker configs
```
