# Руководство по развёртыванию

## Локальное развёртывание

### 1. Установка зависимостей
```bash
# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или: venv\Scripts\activate  # Windows

# Установить зависимости
pip install -r requirements.txt
```

### 2. Запуск API
```bash
# Development
uvicorn app.api.main:app --reload

# Production
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### 3. Docker развёртывание
```bash
# Build
docker-compose build

# Run all services
docker-compose up -d

# Check status
docker-compose ps

# Logs
docker-compose logs -f api
```

## Облачное развёртывание

### AWS
```bash
# Elastic Container Service (ECS)
aws ecs create-service \
    --cluster fashion-ai \
    --service-name api \
    --task-definition fashion-ai:1 \
    --desired-count 2

# GPU instances для ML inference
aws ec2 run-instances \
    --image-id ami-gpu-deep-learning \
    --instance-type g4dn.xlarge
```

### Google Cloud Platform
```bash
# Cloud Run (CPU-only)
gcloud run deploy fashion-ai \
    --image gcr.io/project/fashion-ai \
    --platform managed

# GKE with GPU
gcloud container clusters create fashion-ai \
    --accelerator type=nvidia-tesla-t4,count=1
```

### GPU Cloud Providers
- **Lambda Labs**: Дешёвые GPU
- **RunPod**: Serverless GPU
- **Vast.ai**: Marketplace GPU

## Мониторинг и логирование

### Prometheus + Grafana
```yaml
# docker-compose.yml добавить:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

### Метрики для отслеживания
- Inference latency (p50, p95, p99)
- Model accuracy over time
- Cache hit rate
- GPU utilization
- Error rates

## CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Test
        run: |
          pip install -r requirements.txt
          pytest

  deploy:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: |
          docker build -t fashion-ai:${{ github.sha }}
          docker push fashion-ai:${{ github.sha }}
```

## Производительность

### Оптимизации
1. **Model quantization**: FP16 / INT8
2. **ONNX export**: Для кросс-платформенности
3. **TensorRT**: NVIDIA GPU optimization
4. **Batch inference**: Для высокой нагрузки
5. **Caching**: Redis для частых запросов

### Бенчмарки
```python
# Ожидаемая производительность
Detection (YOLOv8x):    ~30ms на RTX 3090
Classification:         ~15ms на RTX 3090
Embedding (CLIP):       ~50ms на RTX 3090
Full pipeline:          ~150-200ms
```

## Безопасность

### API Security
- Rate limiting (100 req/min)
- API key authentication
- Request validation
- Image size limits (10MB)

### Data Privacy
- Images encrypted at rest
- No persistent storage без согласия
- GDPR compliance
- Data retention policies
