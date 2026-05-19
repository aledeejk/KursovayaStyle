"""
Fashion Item Embeddings using CLIP and FashionCLIP
"""
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer
from PIL import Image
import cv2
from typing import List, Union, Optional
from dataclasses import dataclass
import torch.nn.functional as F


@dataclass
class EmbeddingResult:
    """Embedding result"""
    vector: np.ndarray
    modality: str  # 'image' or 'text'
    dimension: int
    normalized: bool


class FashionEmbeddingModel:
    """
    Multi-modal embedding model for fashion items
    
    Uses CLIP or FashionCLIP for joint image-text embeddings
    """
    
    AVAILABLE_MODELS = {
        'clip-vit-base': 'openai/clip-vit-base-patch32',
        'clip-vit-large': 'openai/clip-vit-large-patch14',
        'fashionclip': 'patrickjohncyh/fashion-clip',
    }
    
    def __init__(
        self,
        model_name: str = 'fashionclip',
        device: str = 'auto',
        normalize: bool = True
    ):
        """
        Initialize embedding model
        
        Args:
            model_name: Model identifier
            device: Device to run on
            normalize: L2-normalize embeddings
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self.normalize = normalize
        
        # Load model
        model_path = self.AVAILABLE_MODELS.get(model_name, model_name)
        
        print(f"Loading {model_name} from {model_path}...")
        
        self.model = CLIPModel.from_pretrained(model_path)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_path)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.projection_dim
        
        print(f"✓ Loaded {model_name}: {self.embedding_dim}D embeddings on {self.device}")
    
    def encode_image(
        self,
        image: Union[np.ndarray, Image.Image, str],
        return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode image to embedding vector
        
        Args:
            image: Image (numpy array, PIL Image, or path)
            return_tensor: Return torch tensor instead of numpy
            
        Returns:
            Embedding vector (512-D for CLIP)
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)
            
            if self.normalize:
                image_features = F.normalize(image_features, p=2, dim=-1)
        
        if return_tensor:
            return image_features
        
        return image_features.cpu().numpy()[0]
    
    def encode_text(
        self,
        text: Union[str, List[str]],
        return_tensor: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode text to embedding vector
        
        Args:
            text: Text string or list of strings
            return_tensor: Return torch tensor instead of numpy
            
        Returns:
            Embedding vector(s)
        """
        # Ensure list
        if isinstance(text, str):
            text = [text]
        
        # Preprocess
        inputs = self.processor(
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Encode
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)
            
            if self.normalize:
                text_features = F.normalize(text_features, p=2, dim=-1)
        
        if return_tensor:
            return text_features if len(text) > 1 else text_features[0]
        
        result = text_features.cpu().numpy()
        return result if len(text) > 1 else result[0]
    
    def encode_batch(
        self,
        images: Optional[List] = None,
        texts: Optional[List[str]] = None,
        batch_size: int = 16
    ) -> Union[List[np.ndarray], tuple]:
        """
        Batch encoding for multiple images/texts
        
        Returns:
            Image embeddings, text embeddings, or both
        """
        image_embeddings = None
        text_embeddings = None
        
        # Encode images
        if images is not None:
            image_embeddings = []
            for i in range(0, len(images), batch_size):
                batch = images[i:i + batch_size]
                
                # Preprocess batch
                inputs = self.processor(images=batch, return_tensors="pt", padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    features = self.model.get_image_features(**inputs)
                    if self.normalize:
                        features = F.normalize(features, p=2, dim=-1)
                
                image_embeddings.extend(features.cpu().numpy())
        
        # Encode texts
        if texts is not None:
            text_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                inputs = self.processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    features = self.model.get_text_features(**inputs)
                    if self.normalize:
                        features = F.normalize(features, p=2, dim=-1)
                
                text_embeddings.extend(features.cpu().numpy())
        
        if images is not None and texts is not None:
            return image_embeddings, text_embeddings
        elif images is not None:
            return image_embeddings
        else:
            return text_embeddings
    
    def compute_similarity(
        self,
        image_embeddings: np.ndarray,
        text_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between image and text embeddings
        
        Returns:
            Similarity matrix (cosine similarity)
        """
        # Ensure 2D
        if len(image_embeddings.shape) == 1:
            image_embeddings = image_embeddings.reshape(1, -1)
        if len(text_embeddings.shape) == 1:
            text_embeddings = text_embeddings.reshape(1, -1)
        
        # Cosine similarity (already normalized)
        similarity = np.dot(image_embeddings, text_embeddings.T)
        
        return similarity
    
    def find_best_matches(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find best matching candidates for query
        
        Returns:
            List of (index, score) tuples
        """
        similarities = self.compute_similarity(
            query_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]
    
    def generate_fashion_description(
        self,
        category: str,
        color: str,
        style: str,
        pattern: str = 'solid'
    ) -> str:
        """Generate descriptive text for fashion item"""
        templates = [
            f"a {color} {style} {category} with {pattern} pattern",
            f"{style} {color} {category}, {pattern} design",
            f"fashionable {color} {category} in {style} style",
        ]
        return templates[0]


class VectorDatabase:
    """
    Simple vector database for fashion item storage and search
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.vectors = []
        self.metadata = []
    
    def add_item(
        self,
        embedding: np.ndarray,
        metadata: dict
    ):
        """Add item to database"""
        self.vectors.append(embedding)
        self.metadata.append(metadata)
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[dict]:
        """
        Search for similar items
        
        Returns:
            List of metadata with similarity scores
        """
        if not self.vectors:
            return []
        
        # Stack all vectors
        candidates = np.stack(self.vectors)
        
        # Compute similarities
        similarities = np.dot(candidates, query_embedding)
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            result = {
                **self.metadata[idx],
                'similarity': float(similarities[idx]),
                'index': int(idx)
            }
            results.append(result)
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = FashionEmbeddingModel(model_name='fashionclip')
    
    # Load image
    image = Image.open("dress.jpg")
    
    # Encode image
    image_emb = model.encode_image(image)
    print(f"Image embedding: {image_emb.shape}")
    
    # Encode text queries
    texts = [
        "elegant red dress",
        "casual blue jeans",
        "formal black suit"
    ]
    text_embs = model.encode_text(texts)
    
    # Find best match
    similarities = model.compute_similarity(image_emb, text_embs)
    print(f"Similarities: {similarities}")
    
    best_match = texts[np.argmax(similarities)]
    print(f"Best match: {best_match}")
