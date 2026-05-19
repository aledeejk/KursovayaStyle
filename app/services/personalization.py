"""
Personalization Service
Персонализация рекомендаций на основе истории пользователя
"""
import json
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


@dataclass
class UserProfile:
    """Профиль пользователя"""
    user_id: str
    preferred_styles: List[str] = None  # casual, formal, business...
    favorite_colors: List[str] = None   # любимые цвета
    disliked_colors: List[str] = None  # не любимые цвета
    size_info: Dict = None             # размеры
    favorite_categories: List[str] = None  # любимые категории одежды
    avoided_categories: List[str] = None   # что не носит
    style_history: List[Dict] = None     # история выборов
    
    def __post_init__(self):
        if self.preferred_styles is None:
            self.preferred_styles = ['casual']
        if self.favorite_colors is None:
            self.favorite_colors = []
        if self.disliked_colors is None:
            self.disliked_colors = []
        if self.size_info is None:
            self.size_info = {}
        if self.favorite_categories is None:
            self.favorite_categories = []
        if self.avoided_categories is None:
            self.avoided_categories = []
        if self.style_history is None:
            self.style_history = []


class PersonalizationService:
    """
    Сервис персонализации
    
    Хранит профили пользователей и адаптирует рекомендации
    """
    
    def __init__(self, storage_path: str = "data/user_profiles"):
        """
        Инициализация
        
        Args:
            storage_path: Путь для хранения профилей
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._cache = {}
    
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """Получить или создать профиль пользователя"""
        if user_id in self._cache:
            return self._cache[user_id]
        
        profile_path = self.storage_path / f"{user_id}.json"
        
        if profile_path.exists():
            with open(profile_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                profile = UserProfile(**data)
        else:
            profile = UserProfile(user_id=user_id)
            self.save_profile(profile)
        
        self._cache[user_id] = profile
        return profile
    
    def save_profile(self, profile: UserProfile):
        """Сохранить профиль"""
        profile_path = self.storage_path / f"{profile.user_id}.json"
        
        with open(profile_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(profile), f, ensure_ascii=False, indent=2)
        
        self._cache[profile.user_id] = profile
    
    def record_outfit_choice(
        self,
        user_id: str,
        outfit: Dict,
        rating: int,  # 1-5
        context: Optional[Dict] = None
    ):
        """
        Записать выбор пользователя
        
        Args:
            user_id: ID пользователя
            outfit: Выбранный outfit
            rating: Оценка 1-5
            context: Контекст (погода, событие и т.д.)
        """
        profile = self.get_or_create_profile(user_id)
        
        # Добавляем в историю
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'outfit': outfit,
            'rating': rating,
            'context': context or {}
        }
        
        profile.style_history.append(history_entry)
        
        # Обновляем предпочтения на основе высоких оценок
        if rating >= 4:
            self._update_preferences_from_choice(profile, outfit)
        
        self.save_profile(profile)
    
    def _update_preferences_from_choice(
        self,
        profile: UserProfile,
        outfit: Dict
    ):
        """Обновить предпочтения на основе выбора"""
        items = outfit.get('items', [])
        
        for item in items:
            # Добавляем цвета в любимые
            color = item.get('color')
            if color and color not in profile.favorite_colors:
                profile.favorite_colors.append(color)
            
            # Добавляем стиль
            style = item.get('style')
            if style and style not in profile.preferred_styles:
                profile.preferred_styles.append(style)
            
            # Добавляем категории
            category = item.get('category')
            if category and category not in profile.favorite_categories:
                profile.favorite_categories.append(category)
    
    def get_personalized_recommendations(
        self,
        user_id: str,
        base_outfits: List[Dict]
    ) -> List[Dict]:
        """
        Отранжировать outfits на основе предпочтений пользователя
        
        Returns:
            Спис outfits с персонализированными score
        """
        profile = self.get_or_create_profile(user_id)
        
        scored_outfits = []
        
        for outfit in base_outfits:
            score = self._calculate_personalization_score(outfit, profile)
            outfit_copy = outfit.copy()
            outfit_copy['personalization_score'] = score
            outfit_copy['personalized_reasoning'] = self._generate_reasoning(outfit, profile)
            scored_outfits.append((score, outfit_copy))
        
        # Сортируем по персонализированному score
        scored_outfits.sort(reverse=True, key=lambda x: x[0])
        
        return [outfit for _, outfit in scored_outfits]
    
    def _calculate_personalization_score(
        self,
        outfit: Dict,
        profile: UserProfile
    ) -> float:
        """Рассчитать score персонализации"""
        score = 0.0
        items = outfit.get('items', [])
        
        if not items:
            return 0.0
        
        for item in items:
            # Цвета
            color = item.get('color', '').lower()
            if color in [c.lower() for c in profile.favorite_colors]:
                score += 0.3
            if color in [c.lower() for c in profile.disliked_colors]:
                score -= 0.5
            
            # Стиль
            style = item.get('style', '').lower()
            if style in [s.lower() for s in profile.preferred_styles]:
                score += 0.2
            
            # Категории
            category = item.get('category', '').lower()
            if category in [c.lower() for c in profile.favorite_categories]:
                score += 0.2
            if category in [c.lower() for c in profile.avoided_categories]:
                score -= 0.4
        
        # Нормализуем
        max_possible = len(items) * 0.7
        if max_possible > 0:
            score = (score / max_possible + 1) / 2  # Приводим к 0-1
        
        return max(0.0, min(1.0, score))
    
    def _generate_reasoning(
        self,
        outfit: Dict,
        profile: UserProfile
    ) -> str:
        """Сгенерировать объяснение персонализации"""
        reasons = []
        items = outfit.get('items', [])
        
        for item in items:
            color = item.get('color', '').lower()
            if color in [c.lower() for c in profile.favorite_colors]:
                reasons.append(f"содержит любимый цвет {color}")
                break
        
        if reasons:
            return f"Рекомендовано, так как {' и '.join(reasons)}"
        
        return "Соответствует вашему стилю"
    
    def update_preferences(
        self,
        user_id: str,
        preferences: Dict
    ):
        """Обновить предпочтения пользователя"""
        profile = self.get_or_create_profile(user_id)
        
        if 'preferred_styles' in preferences:
            profile.preferred_styles = preferences['preferred_styles']
        if 'favorite_colors' in preferences:
            profile.favorite_colors = preferences['favorite_colors']
        if 'disliked_colors' in preferences:
            profile.disliked_colors = preferences['disliked_colors']
        if 'size_info' in preferences:
            profile.size_info = {**profile.size_info, **preferences['size_info']}
        
        self.save_profile(profile)
    
    def get_style_statistics(self, user_id: str) -> Dict:
        """Получить статистику стиля пользователя"""
        profile = self.get_or_create_profile(user_id)
        
        if not profile.style_history:
            return {'message': 'Нет истории'}
        
        # Анализируем историю
        color_counts = {}
        style_counts = {}
        
        for entry in profile.style_history:
            if entry.get('rating', 0) >= 4:  # Только высокие оценки
                outfit = entry.get('outfit', {})
                for item in outfit.get('items', []):
                    color = item.get('color')
                    if color:
                        color_counts[color] = color_counts.get(color, 0) + 1
                    
                    style = item.get('style')
                    if style:
                        style_counts[style] = style_counts.get(style, 0) + 1
        
        return {
            'favorite_colors': sorted(color_counts.items(), key=lambda x: -x[1])[:5],
            'favorite_styles': sorted(style_counts.items(), key=lambda x: -x[1])[:3],
            'total_outfits': len(profile.style_history),
            'high_rated_outfits': sum(1 for e in profile.style_history if e.get('rating', 0) >= 4)
        }


# === COLLABORATIVE FILTERING (простая версия) ===

class CollaborativeFiltering:
    """
    Простой collaborative filtering для outfit recommendations
    
    Находит похожих пользователей и рекомендует то, что понравилось им
    """
    
    def __init__(self, personalization_service: PersonalizationService):
        self.ps = personalization_service
    
    def find_similar_users(
        self,
        target_user_id: str,
        n_similar: int = 5
    ) -> List[str]:
        """
        Найти похожих пользователей по профилю
        
        Returns:
            Список user_id похожих пользователей
        """
        target_profile = self.ps.get_or_create_profile(target_user_id)
        
        similarities = []
        
        # Загружаем все профили
        for profile_file in self.ps.storage_path.glob("*.json"):
            user_id = profile_file.stem
            if user_id == target_user_id:
                continue
            
            profile = self.ps.get_or_create_profile(user_id)
            
            # Считаем сходство
            similarity = self._calculate_similarity(target_profile, profile)
            similarities.append((similarity, user_id))
        
        # Сортируем и возвращаем топ-N
        similarities.sort(reverse=True)
        return [uid for _, uid in similarities[:n_similar]]
    
    def _calculate_similarity(
        self,
        p1: UserProfile,
        p2: UserProfile
    ) -> float:
        """Сходство двух профилей (0-1)"""
        score = 0.0
        
        # Пересечение любимых цветов
        colors1 = set([c.lower() for c in p1.favorite_colors])
        colors2 = set([c.lower() for c in p2.favorite_colors])
        if colors1 and colors2:
            score += len(colors1 & colors2) / max(len(colors1), len(colors2)) * 0.4
        
        # Пересечение стилей
        styles1 = set([s.lower() for s in p1.preferred_styles])
        styles2 = set([s.lower() for s in p2.preferred_styles])
        if styles1 and styles2:
            score += len(styles1 & styles2) / max(len(styles1), len(styles2)) * 0.4
        
        # Пересечение категорий
        cats1 = set([c.lower() for c in p1.favorite_categories])
        cats2 = set([c.lower() for c in p2.favorite_categories])
        if cats1 and cats2:
            score += len(cats1 & cats2) / max(len(cats1), len(cats2)) * 0.2
        
        return score


if __name__ == "__main__":
    # Тест
    ps = PersonalizationService()
    
    # Создаём профиль
    profile = ps.get_or_create_profile("test_user")
    profile.preferred_styles = ['casual', 'business']
    profile.favorite_colors = ['blue', 'black', 'white']
    ps.save_profile(profile)
    
    print(f"Профиль создан: {profile.user_id}")
    print(f"Стилей: {profile.preferred_styles}")
    print(f"Цветов: {profile.favorite_colors}")
