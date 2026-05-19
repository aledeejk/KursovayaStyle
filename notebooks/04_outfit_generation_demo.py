"""
Demo: Outfit Generation
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.outfit_generator import (
    ClothingItem, RuleBasedGenerator, HybridGenerator,
    OutfitStyle, WeatherCondition
)


def main():
    # Создаём тестовый гардероб
    wardrobe = [
        ClothingItem("t1", "top", "t_shirt", "white", "solid", "casual", season=["spring", "summer"]),
        ClothingItem("t2", "top", "shirt", "blue", "striped", "business", season=["all"]),
        ClothingItem("t3", "top", "sweater", "gray", "solid", "casual", season=["fall", "winter"]),
        ClothingItem("b1", "bottom", "jeans", "blue", "solid", "casual", season=["all"]),
        ClothingItem("b2", "bottom", "pants", "black", "solid", "business", season=["all"]),
        ClothingItem("b3", "bottom", "shorts", "beige", "solid", "casual", season=["spring", "summer"]),
        ClothingItem("s1", "shoes", "sneakers", "white", "solid", "casual", season=["all"]),
        ClothingItem("s2", "shoes", "boots", "brown", "solid", "casual", season=["fall", "winter"]),
        ClothingItem("s3", "shoes", "loafers", "black", "solid", "business", season=["all"]),
        ClothingItem("o1", "outerwear", "jacket", "navy", "solid", "casual", season=["spring", "fall"]),
        ClothingItem("o2", "outerwear", "blazer", "gray", "solid", "business", season=["all"]),
    ]
    
    print("="*60)
    print("WARDROBE:")
    for item in wardrobe:
        print(f"  {item.id}: {item.color} {item.subcategory} ({item.style})")
    
    # Rule-based generation
    print("\n" + "="*60)
    print("RULE-BASED OUTFIT GENERATION")
    print("="*60)
    
    rule_generator = RuleBasedGenerator()
    
    # Тест 1: Casual летний образ
    print("\n--- Casual Summer Outfit ---")
    outfits = rule_generator.generate(
        wardrobe, 
        style=OutfitStyle.CASUAL,
        occasion="weekend",
        season="summer"
    )
    
    for i, outfit in enumerate(outfits):
        print(f"\nOutfit {i+1} (score: {outfit.score:.2f}):")
        for item in outfit.items:
            print(f"  - {item.color} {item.subcategory}")
        print(f"  Reasoning: {outfit.reasoning}")
    
    # Тест 2: Business образ
    print("\n--- Business Outfit ---")
    outfits = rule_generator.generate(
        wardrobe,
        style=OutfitStyle.BUSINESS,
        occasion="work",
        season="spring"
    )
    
    for i, outfit in enumerate(outfits):
        print(f"\nOutfit {i+1} (score: {outfit.score:.2f}):")
        for item in outfit.items:
            print(f"  - {item.color} {item.subcategory}")
        print(f"  Reasoning: {outfit.reasoning}")
    
    # Тест 3: Cold weather
    print("\n--- Cold Weather Outfit ---")
    outfits = rule_generator.generate(
        wardrobe,
        style=OutfitStyle.CASUAL,
        occasion="weekend",
        season="winter"
    )
    
    for i, outfit in enumerate(outfits):
        print(f"\nOutfit {i+1} (score: {outfit.score:.2f}):")
        for item in outfit.items:
            print(f"  - {item.color} {item.subcategory}")
        print(f"  Reasoning: {outfit.reasoning}")
    
    # Color harmony test
    print("\n" + "="*60)
    print("COLOR HARMONY TEST")
    print("="*60)
    
    test_items = [
        ClothingItem("1", "top", "shirt", "red"),
        ClothingItem("2", "bottom", "pants", "green"),
    ]
    score = rule_generator._color_harmony_rule(test_items)
    print(f"Red + Green (complementary): {score:.2f}")
    
    test_items = [
        ClothingItem("1", "top", "shirt", "red"),
        ClothingItem("2", "bottom", "pants", "orange"),
    ]
    score = rule_generator._color_harmony_rule(test_items)
    print(f"Red + Orange (similar): {score:.2f}")
    
    test_items = [
        ClothingItem("1", "top", "shirt", "red"),
        ClothingItem("2", "bottom", "pants", "blue"),
        ClothingItem("3", "shoes", "sneakers", "yellow"),
        ClothingItem("4", "accessory", "bag", "purple"),
    ]
    score = rule_generator._color_harmony_rule(test_items)
    print(f"Red + Blue + Yellow + Purple (too many bold): {score:.2f}")
    
    return outfits


if __name__ == "__main__":
    main()
