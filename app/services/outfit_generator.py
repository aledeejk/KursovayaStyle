"""
Outfit Generation Engine

Supports multiple approaches:
1. Rule-based generation
2. ML-based recommendation
3. LLM-based generation
"""
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import random


class OutfitStyle(Enum):
    CASUAL = "casual"
    FORMAL = "formal"
    BUSINESS = "business"
    SPORTY = "sporty"
    VINTAGE = "vintage"
    BOHEMIAN = "bohemian"
    MINIMALIST = "minimalist"
    STREETWEAR = "streetwear"
    EVENING = "evening"


class WeatherCondition(Enum):
    SUNNY = "sunny"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    COLD = "cold"
    HOT = "hot"
    SNOWY = "snowy"


@dataclass
class ClothingItem:
    """Represents a clothing item"""
    id: str
    category: str  # top, bottom, shoes, accessory
    subcategory: str  # t-shirt, jeans, sneakers
    color: str
    pattern: str = "solid"
    style: str = "casual"
    season: List[str] = field(default_factory=lambda: ["spring", "summer", "fall", "winter"])
    embedding: Optional[np.ndarray] = None
    image_url: Optional[str] = None
    attributes: Dict = field(default_factory=dict)


@dataclass
class Outfit:
    """Generated outfit"""
    items: List[ClothingItem]
    style: OutfitStyle
    occasion: str
    weather: Optional[WeatherCondition]
    score: float
    reasoning: str


class RuleBasedGenerator:
    """
    Rule-based outfit generation using fashion principles
    """
    
    # Color harmony rules
    COLOR_WHEEL = {
        'red': 0, 'red-orange': 15, 'orange': 30, 'yellow-orange': 45,
        'yellow': 60, 'yellow-green': 75, 'green': 90, 'blue-green': 105,
        'cyan': 120, 'blue-cyan': 135, 'blue': 180, 'blue-purple': 195,
        'purple': 240, 'red-purple': 255, 'magenta': 300, 'pink': 330,
    }
    
    # Complementary colors (opposite on wheel)
    COMPLEMENTARY = {
        'red': 'green', 'green': 'red',
        'blue': 'orange', 'orange': 'blue',
        'yellow': 'purple', 'purple': 'yellow',
        'black': 'white', 'white': 'black',
    }
    
    # Neutral colors
    NEUTRALS = ['black', 'white', 'gray', 'beige', 'navy', 'brown', 'cream', 'tan']
    
    # Category combinations for outfits
    OUTFIT_TEMPLATES = {
        'casual': [
            ['t_shirt', 'jeans', 'sneakers'],
            ['sweater', 'pants', 'boots'],
            ['hoodie', 'joggers', 'sneakers'],
        ],
        'formal': [
            ['dress_shirt', 'suit_pants', 'dress_shoes', 'blazer'],
            ['blouse', 'skirt', 'heels'],
            ['dress', 'heels', 'clutch'],
        ],
        'business': [
            ['dress_shirt', 'dress_pants', 'loafers', 'blazer'],
            ['blouse', 'pencil_skirt', 'flats'],
        ],
        'sporty': [
            ['tank_top', 'shorts', 'running_shoes'],
            ['athletic_shirt', 'leggings', 'sneakers'],
        ],
    }
    
    def __init__(self):
        self.rules = self._build_rules()
    
    def _build_rules(self) -> Dict:
        """Build fashion rules"""
        return {
            'color_harmony': self._color_harmony_rule,
            'season_appropriate': self._season_rule,
            'occasion_match': self._occasion_rule,
            'fit_balance': self._fit_balance_rule,
        }
    
    def _color_harmony_rule(self, items: List[ClothingItem]) -> float:
        """Score color harmony"""
        if len(items) < 2:
            return 1.0
        
        colors = [item.color.lower() for item in items]
        score = 1.0
        
        # Check for too many bold colors
        bold_colors = [c for c in colors if c not in self.NEUTRALS]
        if len(bold_colors) > 2:
            score -= 0.3 * (len(bold_colors) - 2)
        
        # Check for complementary colors
        for i, c1 in enumerate(colors):
            for c2 in colors[i+1:]:
                if self.COMPLEMENTARY.get(c1) == c2:
                    score += 0.2
        
        return max(0.0, min(1.0, score))
    
    def _season_rule(self, items: List[ClothingItem], season: str) -> float:
        """Check season appropriateness"""
        score = 0.0
        for item in items:
            if season in item.season:
                score += 1.0
        return score / len(items) if items else 0.0
    
    def _occasion_rule(self, items: List[ClothingItem], occasion: str) -> float:
        """Check occasion match"""
        # Map occasion to style
        occasion_styles = {
            'work': ['business', 'formal'],
            'party': ['evening', 'formal'],
            'date': ['casual', 'evening'],
            'gym': ['sporty'],
            'weekend': ['casual', 'bohemian'],
        }
        
        valid_styles = occasion_styles.get(occasion, ['casual'])
        
        score = 0.0
        for item in items:
            if item.style in valid_styles:
                score += 1.0
        
        return score / len(items) if items else 0.0
    
    def _fit_balance_rule(self, items: List[ClothingItem]) -> float:
        """Score fit balance (loose vs tight)"""
        fits = [item.attributes.get('fit', 'regular') for item in items]
        
        loose_count = fits.count('loose') + fits.count('oversized')
        tight_count = fits.count('tight') + fits.count('slim')
        
        # Balanced outfit should have mix
        if loose_count > 0 and tight_count > 0:
            return 1.0
        elif loose_count == len(fits) or tight_count == len(fits):
            return 0.5
        
        return 0.8
    
    def generate(
        self,
        items: List[ClothingItem],
        style: OutfitStyle,
        occasion: str = "casual",
        season: str = "spring",
        weather: Optional[WeatherCondition] = None,
        top_k: int = 3
    ) -> List[Outfit]:
        """
        Generate outfits using rules
        
        Args:
            items: Available clothing items
            style: Desired style
            occasion: Occasion type
            season: Current season
            weather: Weather condition
            top_k: Number of outfits to generate
            
        Returns:
            List of generated outfits
        """
        # Filter by style
        style_items = [item for item in items if item.style == style.value]
        
        if not style_items:
            style_items = items
        
        # Get template for style
        templates = self.OUTFIT_TEMPLATES.get(style.value, self.OUTFIT_TEMPLATES['casual'])
        
        outfits = []
        
        for template in templates[:top_k]:
            outfit_items = []
            
            for category in template:
                # Find matching item
                matching = [
                    item for item in style_items
                    if item.subcategory == category or item.category == category
                ]
                
                if matching:
                    # Score and select best
                    scored = []
                    for item in matching:
                        score = self._score_item(item, outfit_items, season, occasion)
                        scored.append((score, item))
                    
                    scored.sort(reverse=True)
                    outfit_items.append(scored[0][1])
            
            # Score complete outfit
            if len(outfit_items) >= 2:
                total_score = self._score_outfit(outfit_items, season, occasion)
                
                outfit = Outfit(
                    items=outfit_items,
                    style=style,
                    occasion=occasion,
                    weather=weather,
                    score=total_score,
                    reasoning=self._generate_reasoning(outfit_items, total_score)
                )
                outfits.append(outfit)
        
        # Sort by score
        outfits.sort(key=lambda x: x.score, reverse=True)
        
        return outfits[:top_k]
    
    def _score_item(
        self,
        item: ClothingItem,
        current_items: List[ClothingItem],
        season: str,
        occasion: str
    ) -> float:
        """Score individual item for outfit"""
        score = 0.5
        
        # Color harmony with existing items
        if current_items:
            score += self._color_harmony_rule(current_items + [item])
        
        # Season match
        if season in item.season:
            score += 0.3
        
        return score
    
    def _score_outfit(
        self,
        items: List[ClothingItem],
        season: str,
        occasion: str
    ) -> float:
        """Score complete outfit"""
        scores = [
            self._color_harmony_rule(items),
            self._season_rule(items, season),
            self._occasion_rule(items, occasion),
            self._fit_balance_rule(items),
        ]
        
        return np.mean(scores)
    
    def _generate_reasoning(self, items: List[ClothingItem], score: float) -> str:
        """Generate explanation for outfit"""
        colors = [item.color for item in items]
        categories = [item.subcategory for item in items]
        
        reasoning = f"Outfit with {', '.join(categories)}. "
        reasoning += f"Color palette: {', '.join(set(colors))}. "
        
        if score > 0.8:
            reasoning += "Excellent style coherence."
        elif score > 0.6:
            reasoning += "Good balanced look."
        else:
            reasoning += "Casual everyday combination."
        
        return reasoning


class MLBasedGenerator:
    """
    Machine Learning based outfit generation
    
    Uses:
    - Collaborative filtering (similar users)
    - Content-based filtering (item similarity)
    - Neural outfit compatibility
    """
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        # Would load trained models here
        self.compatibility_model = None
    
    def train_compatibility_model(
        self,
        good_outfits: List[List[ClothingItem]],
        bad_outfits: List[List[ClothingItem]]
    ):
        """
        Train outfit compatibility model
        
        Positive examples: good outfits
        Negative examples: random combinations
        """
        # Implementation would use Siamese network or
        # outfit compatibility model
        pass
    
    def score_compatibility(
        self,
        items: List[ClothingItem]
    ) -> float:
        """
        Score outfit compatibility using trained model
        
        Returns:
            Compatibility score 0-1
        """
        if not items or len(items) < 2:
            return 0.0
        
        # Get embeddings
        embeddings = [item.embedding for item in items if item.embedding is not None]
        
        if len(embeddings) < 2:
            return 0.5  # Default score
        
        # Compute pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        # Average similarity as compatibility score
        compatibility = np.mean(similarities)
        
        return float(compatibility)
    
    def generate_embeddings_based(
        self,
        items: List[ClothingItem],
        seed_item: ClothingItem,
        style: OutfitStyle,
        top_k: int = 3
    ) -> List[Outfit]:
        """
        Generate outfits based on embedding similarity
        
        Algorithm:
        1. Start with seed item
        2. Find compatible items based on embedding similarity
        3. Build complete outfit
        """
        if seed_item.embedding is None:
            raise ValueError("Seed item must have embedding")
        
        outfits = []
        
        # Required categories for complete outfit
        required_categories = ['top', 'bottom', 'shoes']
        
        for _ in range(top_k):
            outfit_items = [seed_item]
            
            for category in required_categories:
                if category == seed_item.category:
                    continue
                
                # Find best matching item in category
                category_items = [item for item in items if item.category == category]
                
                if not category_items:
                    continue
                
                # Score by embedding similarity to existing items
                scored = []
                for candidate in category_items:
                    if candidate.embedding is None:
                        continue
                    
                    # Average similarity to current outfit
                    sims = [
                        np.dot(item.embedding, candidate.embedding)
                        for item in outfit_items
                        if item.embedding is not None
                    ]
                    avg_sim = np.mean(sims) if sims else 0
                    scored.append((avg_sim, candidate))
                
                if scored:
                    scored.sort(reverse=True)
                    outfit_items.append(scored[0][1])
            
            # Score outfit
            score = self.score_compatibility(outfit_items)
            
            outfit = Outfit(
                items=outfit_items,
                style=style,
                occasion="auto",
                weather=None,
                score=score,
                reasoning="Generated based on visual similarity"
            )
            outfits.append(outfit)
        
        return outfits


class LLMBasedGenerator:
    """
    LLM-based outfit generation
    
    Uses language models for:
    - Natural language outfit descriptions
    - Style reasoning
    - Personalized recommendations
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        # Would initialize LLM client (OpenAI, Anthropic, etc.)
    
    def generate_with_llm(
        self,
        items: List[ClothingItem],
        user_preferences: Dict,
        context: Dict
    ) -> Outfit:
        """
        Generate outfit using LLM
        
        Args:
            items: Available items
            user_preferences: User's style preferences
            context: Context (weather, occasion, etc.)
            
        Returns:
            Generated outfit with reasoning
        """
        # Build prompt
        prompt = self._build_prompt(items, user_preferences, context)
        
        # Call LLM (pseudo-code)
        # response = llm_client.complete(prompt)
        
        # Parse response to select items
        # For now, return placeholder
        
        selected_items = items[:3]  # Would be parsed from LLM response
        
        outfit = Outfit(
            items=selected_items,
            style=OutfitStyle(user_preferences.get('style', 'casual')),
            occasion=context.get('occasion', 'casual'),
            weather=context.get('weather'),
            score=0.8,
            reasoning="Generated by AI stylist based on your preferences"
        )
        
        return outfit
    
    def _build_prompt(
        self,
        items: List[ClothingItem],
        preferences: Dict,
        context: Dict
    ) -> str:
        """Build prompt for LLM"""
        items_desc = "\n".join([
            f"- {item.id}: {item.color} {item.subcategory} ({item.style})"
            for item in items
        ])
        
        prompt = f"""You are a professional fashion stylist.

Available items:
{items_desc}

User preferences:
- Style: {preferences.get('style', 'casual')}
- Favorite colors: {preferences.get('colors', 'any')}
- Avoid: {preferences.get('avoid', 'none')}

Context:
- Occasion: {context.get('occasion', 'casual')}
- Weather: {context.get('weather', 'sunny')}
- Season: {context.get('season', 'spring')}

Task: Create a stylish outfit by selecting 3-4 items that work well together.
Provide your reasoning.

Response format:
Selected items: [item_ids]
Reasoning: [explanation]
"""
        return prompt


class HybridGenerator:
    """
    Hybrid outfit generator combining all approaches
    """
    
    def __init__(
        self,
        use_rules: bool = True,
        use_ml: bool = True,
        use_llm: bool = False
    ):
        self.rule_generator = RuleBasedGenerator() if use_rules else None
        self.ml_generator = MLBasedGenerator() if use_ml else None
        self.llm_generator = LLMBasedGenerator() if use_llm else None
    
    def generate(
        self,
        items: List[ClothingItem],
        style: OutfitStyle = OutfitStyle.CASUAL,
        occasion: str = "casual",
        season: str = "spring",
        weather: Optional[WeatherCondition] = None,
        user_preferences: Optional[Dict] = None,
        top_k: int = 5,
        seed_item: Optional[ClothingItem] = None
    ) -> List[Outfit]:
        """
        Generate outfits using hybrid approach
        
        Combines results from multiple generators
        """
        all_outfits = []
        
        # Rule-based generation
        if self.rule_generator:
            rule_outfits = self.rule_generator.generate(
                items, style, occasion, season, weather, top_k
            )
            all_outfits.extend(rule_outfits)
        
        # ML-based generation
        if self.ml_generator and seed_item:
            ml_outfits = self.ml_generator.generate_embeddings_based(
                items, seed_item, style, top_k
            )
            all_outfits.extend(ml_outfits)
        
        # LLM-based generation
        if self.llm_generator and user_preferences:
            llm_outfit = self.llm_generator.generate_with_llm(
                items, user_preferences,
                {'occasion': occasion, 'weather': weather, 'season': season}
            )
            all_outfits.append(llm_outfit)
        
        # Deduplicate and rerank
        seen_items = set()
        unique_outfits = []
        
        for outfit in all_outfits:
            key = tuple(sorted([item.id for item in outfit.items]))
            if key not in seen_items:
                seen_items.add(key)
                unique_outfits.append(outfit)
        
        # Sort by score
        unique_outfits.sort(key=lambda x: x.score, reverse=True)
        
        return unique_outfits[:top_k]


# Example usage
if __name__ == "__main__":
    # Create sample items
    items = [
        ClothingItem("1", "top", "t_shirt", "white", "solid", "casual"),
        ClothingItem("2", "bottom", "jeans", "blue", "solid", "casual"),
        ClothingItem("3", "shoes", "sneakers", "white", "solid", "casual"),
        ClothingItem("4", "top", "blouse", "red", "solid", "formal"),
        ClothingItem("5", "bottom", "skirt", "black", "solid", "formal"),
    ]
    
    # Initialize generator
    generator = RuleBasedGenerator()
    
    # Generate outfits
    outfits = generator.generate(
        items,
        style=OutfitStyle.CASUAL,
        occasion="weekend",
        season="summer"
    )
    
    # Display results
    for i, outfit in enumerate(outfits):
        print(f"\nOutfit {i+1} (score: {outfit.score:.2f}):")
        for item in outfit.items:
            print(f"  - {item.color} {item.subcategory}")
        print(f"  Reasoning: {outfit.reasoning}")
