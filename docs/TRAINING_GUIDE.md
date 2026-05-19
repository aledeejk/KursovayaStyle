# Руководство по дообучению моделей

## 1. Дообучение YOLOv8 на Fashion Dataset

### 1.1 Подготовка данных

**DeepFashion2 Dataset**:
```bash
# Скачать с официального сайта
# http://sites.google.com/view/deepfashion2

# Структура датасета:
deepfashion2/
├── train/
│   ├── image/           # 191K изображений
│   └── annos/           # JSON аннотации
├── validation/
│   ├── image/           # 32K изображений
│   └── annos/
└── test/
    └── image/
```

**Конвертация в YOLO формат**:
```python
# convert_deepfashion2_to_yolo.py
import json
import os
from pathlib import Path

def convert_annotation(json_path, output_path, img_width, img_height):
    with open(json_path) as f:
        data = json.load(f)
    
    yolo_lines = []
    
    for item_id, item_data in data.items():
        if 'item1' in item_data:  # Есть одежда
            category_id = item_data['item1']['category_id']  # 1-13
            bbox = item_data['item1']['bounding_box']
            
            # Конвертация в YOLO формат (normalized x_center, y_center, width, height)
            x1, y1, x2, y2 = bbox
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            # category_id в DeepFashion2: 1-13, в YOLO: 0-12
            yolo_class = category_id - 1
            
            yolo_lines.append(
                f"{yolo_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            )
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

# Обработка всех файлов
for json_file in Path('deepfashion2/train/annos').glob('*.json'):
    convert_annotation(
        json_file,
        f"yolo_labels/train/{json_file.stem}.txt",
        img_width=640,
        img_height=640
    )
```

### 1.2 Создание YAML конфигурации

```yaml
# fashion_data.yaml
path: ./deepfashion2_yolo
train: train/images
val: validation/images
test: test/images

# 13 классов DeepFashion2
names:
  0: short_sleeved_shirt      # Футболка/рубашка кор. рукав
  1: long_sleeved_shirt       # Лонгслив/рубашка длин. рукав
  2: short_sleeved_outwear    # Куртка короткий рукав
  3: long_sleeved_outwear     # Пальто/куртка длин. рукав
  4: vest                     # Жилет
  5: sling                    # Топ на бретельках
  6: shorts                   # Шорты
  7: trousers                 # Брюки
  8: skirt                    # Юбка
  9: short_sleeved_dress      # Платье кор. рукав
  10: long_sleeved_dress      # Платье длин. рукав
  11: vest_dress              # Сарафан
  12: sling_dress             # Платье на бретельках
```

### 1.3 Запуск обучения

```python
# train_yolo_fashion.py
from ultralytics import YOLO

# Загружаем предобученную модель
model = YOLO('yolov8n.pt')  # или yolov8s.pt, yolov8m.pt для лучшего качества

# Обучение
results = model.train(
    data='fashion_data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0,  # GPU
    patience=20,  # Early stopping
    save=True,
    project='fashion_training',
    name='yolov8_fashion',
    
    # Аугментации
    hsv_h=0.015,  # Hue
    hsv_s=0.7,    # Saturation
    hsv_v=0.4,    # Value
    degrees=5,    # Rotation
    translate=0.1,
    scale=0.5,
    shear=2,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    mixup=0.1,
)

# Сохранение лучшей модели
best_model = YOLO('fashion_training/yolov8_fashion/weights/best.pt')
```

### 1.4 Валидация

```python
# Валидация на тестовом наборе
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Тест на одном изображении
results = model('test_image.jpg')
results[0].show()
```

## 2. Двухэтапный подход (альтернатива)

Если нет возможности дообучить YOLO, используйте двухэтапный подход:

### Этап 1: Детекция людей
```python
from ultralytics import YOLO

# Стандартная YOLO для детекции людей
person_detector = YOLO('yolov8n.pt')
results = person_detector(image, classes=[0])  # class 0 = person
```

### Этап 2: Классификация одежды
```python
import torch
from torchvision import transforms
from PIL import Image

# Загружаем классификатор на EfficientNet
classifier = torch.load('efficientnet_fashion_classifier.pth')
classifier.eval()

# Преобразования
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Классификация crop'а
for person_bbox in detected_people:
    crop = image[y1:y2, x1:x2]
    
    # Делим на верх/низ (эвристика)
    h = crop.shape[0]
    top_half = crop[:h//2, :]
    bottom_half = crop[h//2:, :]
    
    # Классифицируем
    top_tensor = transform(Image.fromarray(top_half)).unsqueeze(0)
    bottom_tensor = transform(Image.fromarray(bottom_half)).unsqueeze(0)
    
    with torch.no_grad():
        top_class = classifier(top_tensor)
        bottom_class = classifier(bottom_tensor)
    
    # Результаты
    clothing_items = [
        {'category': 'top', 'class': top_class},
        {'category': 'bottom', 'class': bottom_class}
    ]
```

## 3. Русскоязычные и специфические датасеты

### 3.1 Создание собственного датасета

**Для национальной одежды или зимних вещей**:

```
my_fashion_dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt (YOLO format)
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── data.yaml
```

**Аннотация**:
- Используйте LabelImg: `pip install labelImg`
- Или CVAT: https://cvat.org
- Roboflow: https://roboflow.com (рекомендуется)

### 3.2 Пример data.yaml для русскоязычных категорий

```yaml
path: ./russian_fashion
train: train/images
val: val/images

names:
  0: варенка           # Валенки, угги
  1: пуховик           # Зимняя куртка
  2: шуба              # Меховая шуба
  3: платок            # Шерстяной платок
  4: валенки           # Русские валенки
  5: косоворотка       # Традиционная рубашка
  6: сарафан_трад      # Традиционный сарафан
  7: кардиган          # Вязаный кардиган
  8: водолазка         # Гольф
  9: дубленка          # Дублёнка
```

### 3.3 Аугментация для зимней одежды

```python
# Добавляем специфичные аугментации для зимы
import albumentations as A

transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.GaussNoise(var_limit=(10, 50), p=0.3),  # Имитация снега
    A.RandomSnow(snow_point_lower=0.1, snow_point_upper=0.3, p=0.3),
    A.CLAHE(clip_limit=4.0, p=0.3),  # Контраст для тёмной зимней одежды
])
```

## 4. Дообучение FashionCLIP

```python
# train_fashionclip.py
from transformers import CLIPProcessor, CLIPModel, TrainingArguments, Trainer
from datasets import load_dataset

# Загружаем FashionCLIP
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
processor = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")

# Датасет пар (image, text)
dataset = load_dataset("your_fashion_captions_dataset")

# Fine-tuning
training_args = TrainingArguments(
    output_dir="./fashionclip_ft",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
```

## 5. Рекомендуемые ресурсы

### Готовые модели

1. **YOLOv8 Fashion (DeepFashion2)**
   - Искать на: https://github.com/switchablenorms/DeepFashion2/issues
   - Или дообучить самостоятельно по инструкции выше

2. **Hugging Face**
   - `valentinafeve/yolov8_fashion`
   - `patrickjohncyh/fashion-clip`
   
3. **Roboflow Universe**
   - https://universe.roboflow.com
   - Поиск: "fashion", "clothing", "apparel"

### Инструменты разметки

- **CVAT**: https://cvat.org (бесплатный, open-source)
- **Label Studio**: https://labelstud.io
- **Roboflow**: https://roboflow.com (разметка + аугментация)
- **LabelImg**: Локальный инструмент

## 6. Чеклист обучения

- [ ] Скачать DeepFashion2 или подготовить свой датасет
- [ ] Конвертировать аннотации в YOLO формат
- [ ] Создать data.yaml
- [ ] Запустить обучение (100+ эпох)
- [ ] Валидировать на тестовом наборе (mAP50 > 0.6)
- [ ] Экспорт в ONNX/TensorRT (опционально)
- [ ] Заменить `model_path` в `fashion_detector.py`

## 7. Ожидаемые результаты

| Модель | Dataset | mAP50 | Размер | Inference |
|--------|---------|-------|--------|-----------|
| YOLOv8n | DeepFashion2 | 0.65 | 6MB | 10ms |
| YOLOv8s | DeepFashion2 | 0.72 | 23MB | 15ms |
| YOLOv8m | DeepFashion2 | 0.78 | 54MB | 25ms |
| YOLOv8l | DeepFashion2 | 0.82 | 90MB | 40ms |

**Минимально для production**: YOLOv8s (баланс скорость/качество)
