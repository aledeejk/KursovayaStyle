# Модели для Fashion AI System

## 1. Detection модели

### YOLOv8 (рекомендуется)
- **Версии**: nano, small, medium, large, xlarge
- **Рекомендация**: YOLOv8x для best quality, YOLOv8m для скорость/качество
- **Преимущества**:
  - Быстрый inference
  - Хорошая точность
  - Легкий fine-tuning
- **Установка**: `pip install ultralytics`

### Faster R-CNN
- **База**: ResNet-50/101 FPN
- **Использование**: Когда нужна высокая точность, скорость не критична
- **PyTorch**: `torchvision.models.detection.fasterrcnn_resnet50_fpn`

### RT-DETR
- **Новая модель** от Baidu
- **Баланс**: Между speed и accuracy
- **Особенности**: Transformer-based

## 2. Classification модели

### EfficientNet (рекомендуется)
- **Версии**: B0-B7
- **Рекомендация**: B4 для production (хороший баланс)
- **Преимущества**:
  - Compound scaling
  - State-of-the-art accuracy
  - Efficient
- **Установка**: `pip install timm`

### ConvNeXt
- **Альтернатива**: Modern CNN архитектура
- **Преимущества**: Better than Swin Transformer на некоторых задачах

### Swin Transformer
- **Архитектура**: Hierarchical vision transformer
- **Использование**: Когда нужна максимальная accuracy

## 3. Embedding модели

### FashionCLIP (рекомендуется)
- **URL**: `patrickjohncyh/fashion-clip`
- **Особенности**: Специально обучен на fashion data
- **Задачи**: 
  - Image-text retrieval
  - Zero-shot classification
  - Outfit compatibility

### CLIP (OpenAI)
- **Модели**: ViT-B/32, ViT-L/14
- **Преимущества**: Общий vision-language understanding
- **Fine-tuning**: Можно адаптировать под fashion

### ALIGN
- **Google**: Similar to CLIP
- **Датасет**: 1.8B image-text pairs

## 4. Outfit Generation модели

### Compatibility Prediction
- **Siamese Networks**: Для pairwise compatibility
- **Graph Neural Networks**: Outfit как граф
- **Transformer**: Self-attention между items

### Neural Outfit Models
```python
# Пример архитектуры
class OutfitCompatibilityModel(nn.Module):
    def __init__(self, embedding_dim=512):
        self.encoder = nn.TransformerEncoder(...)
        self.compatibility_head = nn.Linear(embedding_dim, 1)
    
    def forward(self, item_embeddings):
        # Self-attention между items
        encoded = self.encoder(item_embeddings)
        # Compatibility score
        score = self.compatibility_head(encoded.mean(dim=0))
        return torch.sigmoid(score)
```

## 5. Pre-trained weights

### Где скачать:
1. **Hugging Face Hub**: `huggingface.co`
2. **PyTorch Hub**: `pytorch.org/hub`
3. **Timm**: `rwightman/pytorch-image-models`
4. **Ultralytics**: Встроенная загрузка

### Пример загрузки:
```python
# YOLOv8
from ultralytics import YOLO
model = YOLO('yolov8x.pt')

# EfficientNet
import timm
model = timm.create_model('efficientnet_b4', pretrained=True)

# FashionCLIP
from transformers import CLIPModel
model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip')
```

## 6. Выбор моделей для задач

| Задача | Модель | Размер | Speed | Accuracy |
|--------|--------|--------|-------|----------|
| Detection | YOLOv8m | 50MB | Fast | Good |
| Detection | YOLOv8x | 130MB | Medium | Best |
| Classification | EfficientNet-B4 | 75MB | Medium | Very Good |
| Classification | EfficientNet-B0 | 20MB | Fast | Good |
| Embeddings | FashionCLIP | 600MB | Medium | Best |
| Embeddings | CLIP-ViT-B/32 | 400MB | Fast | Good |
