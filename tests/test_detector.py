"""
Tests for Clothing Detector
"""
import pytest
import numpy as np
import cv2
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.clothing_detector import ClothingDetector, DetectionResult


class TestClothingDetector:
    """Test cases for clothing detector"""
    
    @pytest.fixture
    def detector(self):
        return ClothingDetector(
            model_path="yolov8n.pt",  # Small model for tests
            conf_threshold=0.3,
            device="cpu"
        )
    
    @pytest.fixture
    def sample_image(self):
        """Create sample test image"""
        img = np.ones((640, 480, 3), dtype=np.uint8) * 200
        # Draw rectangle simulating a person
        cv2.rectangle(img, (100, 50), (400, 600), (100, 150, 200), -1)
        return img
    
    def test_detector_initialization(self, detector):
        """Test detector initializes correctly"""
        assert detector.device in ["cpu", "cuda"]
        assert detector.conf_threshold == 0.3
        assert detector.model is not None
    
    def test_detection_result_structure(self):
        """Test DetectionResult dataclass"""
        result = DetectionResult(
            bbox=(100, 100, 200, 200),
            confidence=0.85,
            class_id=0,
            class_name="person",
            cropped_image=None
        )
        
        assert result.bbox == (100, 100, 200, 200)
        assert result.confidence == 0.85
        assert result.class_name == "person"
    
    def test_detect_returns_list(self, detector, sample_image):
        """Test detect returns list of results"""
        results = detector.detect(sample_image)
        assert isinstance(results, list)
    
    def test_batch_detection(self, detector, sample_image):
        """Test batch processing"""
        images = [sample_image] * 3
        results = detector.detect_batch(images, batch_size=2)
        
        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)


class TestPreprocessing:
    """Test image preprocessing"""
    
    def test_bgr_to_rgb_conversion(self):
        """Test color space conversion"""
        bgr = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        
        # First pixel: BGR -> RGB
        assert bgr[0, 0, 0] == rgb[0, 0, 2]  # B -> R
        assert bgr[0, 0, 1] == rgb[0, 0, 1]  # G -> G
        assert bgr[0, 0, 2] == rgb[0, 0, 0]  # R -> B


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
