"""
Clothing Classification using EfficientNet
"""
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import timm


@dataclass
class ClassificationResult:
    """Classification result"""
    category: str
    confidence: float
    all_scores: Dict[str, float]
    attributes: Dict[str, str]  # color, pattern, style, etc.


class ClothingClassifier:
    """
    Multi-task clothing classifier
    
    Tasks:
    - Category classification (dress, shirt, pants, etc.)
    - Color recognition
    - Pattern detection
    - Style classification (casual, formal, sporty)
    """
    
    # Fashion categories
    CATEGORIES = [
        't_shirt', 'shirt', 'sweater', 'cardigan',
        'jacket', 'coat', 'dress', 'skirt',
        'pants', 'shorts', 'jeans', 'leggings',
        'suit', 'blazer', 'vest', 'overalls',
        'jumpsuit', 'romper', 'swimwear', 'lingerie'
    ]
    
    # Colors
    COLORS = [
        'black', 'white', 'gray', 'red', 'blue',
        'green', 'yellow', 'orange', 'purple', 'pink',
        'brown', 'beige', 'navy', 'burgundy', 'teal'
    ]
    
    # Patterns
    PATTERNS = [
        'solid', 'striped', 'checkered', 'floral',
        'polka_dot', 'geometric', 'animal_print', 'camouflage'
    ]
    
    # Styles
    STYLES = [
        'casual', 'formal', 'business', 'sporty',
        'vintage', 'bohemian', 'minimalist', 'streetwear'
    ]
    
    def __init__(
        self,
        model_name: str = "efficientnet_b4",
        num_categories: int = 20,
        num_colors: int = 15,
        num_patterns: int = 8,
        num_styles: int = 8,
        device: str = "auto",
        pretrained: bool = True
    ):
        """
        Initialize classifier
        
        Args:
            model_name: TIMM model name
            num_categories: Number of clothing categories
            device: Device to run on
            pretrained: Use pretrained weights
        """
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Build model
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove default head
        )
        
        # Get feature dimension
        self.feature_dim = self.model.num_features
        
        # Multi-task heads
        self.category_head = nn.Linear(self.feature_dim, num_categories)
        self.color_head = nn.Linear(self.feature_dim, num_colors)
        self.pattern_head = nn.Linear(self.feature_dim, num_patterns)
        self.style_head = nn.Linear(self.feature_dim, num_styles)
        
        # Move to device
        self.model = self.model.to(self.device)
        self.category_head = self.category_head.to(self.device)
        self.color_head = self.color_head.to(self.device)
        self.pattern_head = self.pattern_head.to(self.device)
        self.style_head = self.style_head.to(self.device)
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.model.eval()
        print(f"✓ Loaded classifier: {model_name} on {self.device}")
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for inference"""
        # Convert BGR to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            if image.dtype == np.uint8:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        pil_image = Image.fromarray(image)
        tensor = self.transform(pil_image)
        return tensor.unsqueeze(0).to(self.device)
    
    @torch.no_grad()
    def classify(self, image: np.ndarray) -> ClassificationResult:
        """
        Classify clothing item
        
        Args:
            image: Cropped clothing image
            
        Returns:
            ClassificationResult with all predictions
        """
        # Preprocess
        x = self.preprocess(image)
        
        # Extract features
        features = self.model(x)
        
        # Get predictions for each task
        category_logits = self.category_head(features)
        color_logits = self.color_head(features)
        pattern_logits = self.pattern_head(features)
        style_logits = self.style_head(features)
        
        # Softmax
        category_probs = torch.softmax(category_logits, dim=1)[0]
        color_probs = torch.softmax(color_logits, dim=1)[0]
        pattern_probs = torch.softmax(pattern_logits, dim=1)[0]
        style_probs = torch.softmax(style_logits, dim=1)[0]
        
        # Get top predictions
        category_idx = torch.argmax(category_probs).item()
        color_idx = torch.argmax(color_probs).item()
        pattern_idx = torch.argmax(pattern_probs).item()
        style_idx = torch.argmax(style_probs).item()
        
        # Build result
        all_scores = {
            self.CATEGORIES[i]: category_probs[i].item()
            for i in range(len(self.CATEGORIES))
        }
        
        attributes = {
            'color': self.COLORS[color_idx],
            'pattern': self.PATTERNS[pattern_idx],
            'style': self.STYLES[style_idx],
            'color_confidence': f"{color_probs[color_idx]:.2f}",
            'pattern_confidence': f"{pattern_probs[pattern_idx]:.2f}",
            'style_confidence': f"{style_probs[style_idx]:.2f}"
        }
        
        return ClassificationResult(
            category=self.CATEGORIES[category_idx],
            confidence=category_probs[category_idx].item(),
            all_scores=all_scores,
            attributes=attributes
        )
    
    def classify_batch(
        self,
        images: List[np.ndarray],
        batch_size: int = 8
    ) -> List[ClassificationResult]:
        """Batch classification"""
        results = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            tensors = torch.cat([self.preprocess(img) for img in batch])
            
            with torch.no_grad():
                features = self.model(tensors)
                
                category_logits = self.category_head(features)
                category_probs = torch.softmax(category_logits, dim=1)
                
                for j, probs in enumerate(category_probs):
                    idx = torch.argmax(probs).item()
                    results.append(ClassificationResult(
                        category=self.CATEGORIES[idx],
                        confidence=probs[idx].item(),
                        all_scores={},
                        attributes={}
                    ))
        
        return results


class SimpleAttributeClassifier:
    """
    Simplified attribute classifier using pre-trained models
    """
    
    def __init__(self, device: str = "auto"):
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
    
    def extract_color(self, image: np.ndarray) -> Dict[str, any]:
        """Extract dominant colors from image"""
        # Convert to RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Reshape for clustering
        pixels = image_rgb.reshape(-1, 3).astype(np.float32)
        
        # K-means clustering to find dominant colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Count frequencies
        _, counts = np.unique(labels, return_counts=True)
        
        # Sort by frequency
        sorted_indices = np.argsort(counts)[::-1]
        
        dominant_colors = []
        for idx in sorted_indices[:3]:
            color = centers[idx].astype(int)
            percentage = counts[idx] / len(pixels) * 100
            dominant_colors.append({
                'rgb': tuple(color),
                'percentage': round(percentage, 2)
            })
        
        return {
            'dominant_colors': dominant_colors,
            'primary_color': dominant_colors[0]['rgb'] if dominant_colors else None
        }
    
    def classify_texture(self, image: np.ndarray) -> Dict[str, float]:
        """Classify texture/pattern using simple features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture features
        # 1. Standard deviation (indicates variation)
        std = np.std(gray)
        
        # 2. Edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # 3. Local binary pattern variance
        # (Simplified version)
        
        # Heuristic classification
        scores = {
            'solid': 0.0,
            'striped': 0.0,
            'patterned': 0.0,
            'textured': 0.0
        }
        
        if std < 30:
            scores['solid'] = 0.9
        elif edge_density > 0.1:
            scores['patterned'] = 0.8
        elif std > 60:
            scores['textured'] = 0.7
        else:
            scores['striped'] = 0.6
        
        return scores


# Example usage
if __name__ == "__main__":
    import cv2
    
    # Initialize classifier
    classifier = ClothingClassifier()
    
    # Load test image
    image = cv2.imread("clothing_item.jpg")
    
    # Classify
    result = classifier.classify(image)
    
    print(f"Category: {result.category} ({result.confidence:.2f})")
    print(f"Attributes: {result.attributes}")
