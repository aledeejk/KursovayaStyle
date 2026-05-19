# Рекомендуемые датасеты

## Для обучения детектора

### 1. DeepFashion2
- **URL**: https://github.com/switchablenorms/DeepFashion2
- **Размер**: 491K изображений
- **Классы**: 13 категорий одежды
- **Аннотации**: bbox, segmentation, landmarks, pairs
- **Использование**: Основной датасет для fashion detection

### 2. Fashionpedia
- **URL**: https://fashionpedia.github.io/home/
- **Размер**: 48K изображений
- **Классы**: 27 категорий, 294 атрибута
- **Аннотации**: Fine-grained categories + attributes
- **Использование**: Детальная классификация

### 3. COCO + Fashion добавления
- COCO имеет ограниченные fashion классы
- Дополнить DeepFashion2 для лучшей генерализации

## Для классификации

### 1. DeepFashion (Category and Attribute Prediction)
- **Классы**: 50 категорий
- **Атрибуты**: 1000+ атрибутов
- **Задачи**: Category, attribute prediction

### 2. Fashion-MNIST
- **Размер**: 70K изображений
- **Классы**: 10 категорий
- **Использование**: Базовые эксперименты, не для production

### 3. CelebA (для атрибутов)
- **Атрибуты**: 40 binary attributes
- **Использование**: Transfer learning для атрибутов

## Для embeddings / retrieval

### 1. Street2Shop
- **URL**: http://www.tamaraberg.com/street2shop/
- **Задача**: Street photo → Shop items
- **Использование**: Cross-domain retrieval

### 2. DARN / (Exact Street to Shop)
- **Размер**: 200K+ пар
- **Задача**: Find exact product match

### 3. Polyvore Outfits
- **URL**: https://github.com/xiaoxiaoluoafeng/polyvore-outfits
- **Задача**: Outfit compatibility prediction
- **Использование**: Outfit generation training

### 4. FashionGen
- **URL**: https://fashion-gen.com/
- **Задача**: Text-to-image, image-to-text
- **Использование**: Мультимодальные embeddings

## Для outfit generation

### 1. Polyvore (Polyvore Outfits Dataset)
- **Размер**: 68K outfits, 365K items
- **Аннотации**: Outfit композиции, likes, descriptions
- **Форматы**: JSON, images
- **Задачи**: 
  - Fill-in-the-blank
  - Compatibility prediction
  - Outfit generation

### 2. Maryland Polyvore
- **Размер**: 33K outfits
- **Аннотации**: Fine-grained categories, titles

### 3. IQON (Japanese fashion)
- **Размер**: 1M+ outfits
- **Особенности**: Богатые метаданные

## Самоподготовленные данные

### Структура для разметки:
```
dataset/
├── images/
│   ├── 001.jpg
│   ├── 002.jpg
│   └── ...
├── annotations/
│   ├── 001.json
│   ├── 002.json
│   └── ...
└── metadata.csv
```

### Формат аннотации:
```json
{
  "image_id": "001",
  "items": [
    {
      "bbox": [100, 200, 300, 400],
      "category": "dress",
      "color": "red",
      "pattern": "solid",
      "style": "formal"
    }
  ]
}
```

## Рекомендуемый подход

### Phase 1: Использование pretrained
1. **Detection**: YOLOv8 pretrained on COCO
2. **Classification**: EfficientNet pretrained on ImageNet
3. **Embeddings**: FashionCLIP (ready-to-use)
4. **Outfits**: Rule-based + Polyvore data

### Phase 2: Fine-tuning
1. Fine-tune detector on DeepFashion2
2. Fine-tune classifier на своих данных
3. Fine-tune embeddings на fashion-specific data
4. Train compatibility model on Polyvore

### Phase 3: Production
- Ensemble моделей
- Active learning для улучшения
- User feedback loop
