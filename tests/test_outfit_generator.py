"""
Tests for Outfit Generator
"""
import pytest
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.outfit_generator import (
    ClothingItem, RuleBasedGenerator, HybridGenerator,
    OutfitStyle, Outfit
)


class TestClothingItem:
    """Test ClothingItem dataclass"""
    
    def test_item_creation(self):
        """Test creating clothing item"""
        item = ClothingItem(
            id="test_1",
            category="top",
            subcategory="t_shirt",
            color="blue",
            pattern="solid",
            style="casual"
        )
        
        assert item.id == "test_1"
        assert item.category == "top"
        assert item.color == "blue"
    
    def test_item_with_embedding(self):
        """Test item with embedding vector"""
        embedding = np.random.randn(512)
        item = ClothingItem(
            id="test_2",
            category="bottom",
            subcategory="jeans",
            color="black",
            embedding=embedding
        )
        
        assert item.embedding is not None
        assert item.embedding.shape == (512,)


class TestRuleBasedGenerator:
    """Test rule-based outfit generation"""
    
    @pytest.fixture
    def generator(self):
        return RuleBasedGenerator()
    
    @pytest.fixture
    def sample_wardrobe(self):
        """Create sample wardrobe for testing"""
        return [
            ClothingItem("t1", "top", "t_shirt", "white", "solid", "casual"),
            ClothingItem("t2", "top", "shirt", "blue", "striped", "business"),
            ClothingItem("b1", "bottom", "jeans", "blue", "solid", "casual"),
            ClothingItem("b2", "bottom", "pants", "black", "solid", "business"),
            ClothingItem("s1", "shoes", "sneakers", "white", "solid", "casual"),
        ]
    
    def test_generator_initialization(self, generator):
        """Test generator initializes with rules"""
        assert 'color_harmony' in generator.rules
        assert 'season_appropriate' in generator.rules
    
    def test_color_harmony_rule(self, generator):
        """Test color harmony scoring"""
        # Complementary colors (should score well)
        items = [
            ClothingItem("1", "top", "shirt", "red"),
            ClothingItem("2", "bottom", "pants", "green"),
        ]
        score = generator._color_harmony_rule(items)
        assert 0 <= score <= 1
        assert score > 0.5  # Complementary should score well
    
    def test_too_many_bold_colors(self, generator):
        """Test penalty for too many bold colors"""
        items = [
            ClothingItem("1", "top", "shirt", "red"),
            ClothingItem("2", "bottom", "pants", "blue"),
            ClothingItem("3", "shoes", "sneakers", "yellow"),
            ClothingItem("4", "accessory", "bag", "purple"),
        ]
        score = generator._color_harmony_rule(items)
        assert score < 0.8  # Should be penalized
    
    def test_outfit_generation(self, generator, sample_wardrobe):
        """Test complete outfit generation"""
        outfits = generator.generate(
            sample_wardrobe,
            style=OutfitStyle.CASUAL,
            occasion="weekend",
            season="spring"
        )
        
        assert len(outfits) > 0
        assert all(isinstance(o, Outfit) for o in outfits)
        assert all(len(o.items) >= 2 for o in outfits)
    
    def test_outfit_scoring(self, generator, sample_wardrobe):
        """Test outfit has valid score"""
        outfits = generator.generate(
            sample_wardrobe,
            style=OutfitStyle.CASUAL
        )
        
        if outfits:
            assert 0 <= outfits[0].score <= 1
            assert outfits[0].reasoning is not None


class TestHybridGenerator:
    """Test hybrid generator combining multiple approaches"""
    
    def test_hybrid_initialization(self):
        """Test hybrid generator with different configs"""
        gen1 = HybridGenerator(use_rules=True, use_ml=False, use_llm=False)
        assert gen1.rule_generator is not None
        
        gen2 = HybridGenerator(use_rules=False, use_ml=True, use_llm=False)
        assert gen2.ml_generator is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
