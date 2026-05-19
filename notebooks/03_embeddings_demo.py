"""
Demo: Fashion Embeddings
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.embeddings import FashionEmbeddingModel


def main():
    # Инициализация модели
    print("Loading FashionCLIP...")
    model = FashionEmbeddingModel(model_name='fashionclip')
    
    # Создаём тестовые изображения (в реальности - загрузка)
    from PIL import Image
    
    # Создаём простые тестовые изображения
    img1 = Image.new('RGB', (224, 224), color='red')
    img2 = Image.new('RGB', (224, 224), color='blue')
    img3 = Image.new('RGB', (224, 224), color='darkred')
    
    # Энкодинг изображений
    print("\nEncoding images...")
    emb1 = model.encode_image(img1)
    emb2 = model.encode_image(img2)
    emb3 = model.encode_image(img3)
    
    print(f"Embedding 1 shape: {emb1.shape}")
    print(f"Embedding 2 shape: {emb2.shape}")
    
    # Энкодинг текстовых описаний
    print("\nEncoding text queries...")
    texts = [
        "red dress",
        "blue jeans",
        "red casual wear"
    ]
    text_embs = model.encode_text(texts)
    
    # Поиск похожих
    print("\nSimilarity scores:")
    
    # Image-text similarity
    sim_matrix = model.compute_similarity(
        np.stack([emb1, emb2, emb3]),
        np.stack(text_embs)
    )
    
    print("\nImage-to-Text Similarity:")
    for i, img_name in enumerate(['Red', 'Blue', 'DarkRed']):
        print(f"\n{img_name} image:")
        for j, text in enumerate(texts):
            score = sim_matrix[i, j]
            bar = "█" * int(score * 20)
            print(f"  vs '{text}': {score:.3f} {bar}")
    
    # Image-image similarity
    print("\nImage-to-Image Similarity:")
    img_sim = model.compute_similarity(
        np.stack([emb1, emb2, emb3]),
        np.stack([emb1, emb2, emb3])
    )
    print(img_sim)
    
    # Find best matches
    print("\nBest text match for Red image:")
    best_matches = model.find_best_matches(emb1, np.stack(text_embs), top_k=3)
    for idx, score in best_matches:
        print(f"  {texts[idx]}: {score:.3f}")
    
    return emb1, emb2, text_embs


if __name__ == "__main__":
    main()
