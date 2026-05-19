"""
Color Analysis Module
Извлечение цветовой палитры из одежды и цветовая гармония
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from collections import Counter


@dataclass
class ColorInfo:
    """Информация о цвете"""
    rgb: Tuple[int, int, int]
    hex: str
    percentage: float
    color_name: Optional[str] = None


class ColorAnalyzer:
    """
    Анализатор цветов одежды
    
    Использует K-Means clustering для поиска доминирующих цветов
    """
    
    # Базовые названия цветов для русского языка
    COLOR_NAMES_RU = {
        'black': 'чёрный',
        'white': 'белый',
        'gray': 'серый',
        'red': 'красный',
        'blue': 'синий',
        'green': 'зелёный',
        'yellow': 'жёлтый',
        'orange': 'оранжевый',
        'purple': 'фиолетовый',
        'pink': 'розовый',
        'brown': 'коричневый',
        'beige': 'бежевый',
        'navy': 'тёмно-синий',
        'burgundy': 'бордовый',
        'teal': 'бирюзовый',
        'cream': 'кремовый',
        'tan': 'загар',
    }
    
    # Цветовой круг для гармонии (Hue values 0-180 в OpenCV)
    COLOR_WHEEL = {
        'red': 0,
        'orange': 15,
        'yellow': 30,
        'green': 60,
        'cyan': 90,
        'blue': 120,
        'purple': 150,
        'magenta': 165,
    }
    
    def __init__(self, n_colors: int = 5):
        """
        Инициализация
        
        Args:
            n_colors: Количество доминирующих цветов для извлечения
        """
        self.n_colors = n_colors
    
    def extract_dominant_colors(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> List[ColorInfo]:
        """
        Извлечь доминирующие цвета из изображения
        
        Args:
            image: BGR изображение (OpenCV)
            mask: Маска для учёта только определённых пикселей
            
        Returns:
            Список ColorInfo с RGB, hex, percentage
        """
        # Конвертация в RGB
        if len(image.shape) == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Применяем маску если есть
        if mask is not None:
            image_rgb = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
        
        # Reshape для K-Means
        pixels = image_rgb.reshape(-1, 3)
        
        # Убираем чёрные пиксели (фон)
        pixels = pixels[(pixels.sum(axis=1) > 30) & (pixels.sum(axis=1) < 750)]
        
        if len(pixels) == 0:
            return []
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=min(self.n_colors, len(pixels)), random_state=42, n_init=10)
        kmeans.fit(pixels)
        
        # Подсчёт частот
        labels = kmeans.labels_
        counts = Counter(labels)
        total = sum(counts.values())
        
        # Формируем результат
        colors = []
        for idx, center in enumerate(kmeans.cluster_centers_):
            rgb = tuple(map(int, center))
            hex_color = '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])
            percentage = (counts[idx] / total) * 100
            
            color_name = self._rgb_to_name(rgb)
            
            colors.append(ColorInfo(
                rgb=rgb,
                hex=hex_color,
                percentage=round(percentage, 2),
                color_name=color_name
            ))
        
        # Сортировка по проценту
        colors.sort(key=lambda x: x.percentage, reverse=True)
        
        return colors
    
    def extract_primary_color(self, image: np.ndarray) -> ColorInfo:
        """Извлечь основной цвет (самый частый)"""
        colors = self.extract_dominant_colors(image, n_colors=1)
        return colors[0] if colors else ColorInfo(rgb=(128, 128, 128), hex="#808080", percentage=100.0)
    
    def _rgb_to_name(self, rgb: Tuple[int, int, int]) -> str:
        """Конвертировать RGB в название цвета"""
        # Простое сопоставление по ближайшему известному цвету
        r, g, b = rgb
        
        # Определение по диапазонам
        if r < 50 and g < 50 and b < 50:
            return self.COLOR_NAMES_RU['black']
        elif r > 200 and g > 200 and b > 200:
            return self.COLOR_NAMES_RU['white']
        elif abs(r - g) < 20 and abs(g - b) < 20:
            return self.COLOR_NAMES_RU['gray']
        elif r > 150 and g < 100 and b < 100:
            return self.COLOR_NAMES_RU['red']
        elif b > 150 and r < 100 and g < 100:
            return self.COLOR_NAMES_RU['blue']
        elif g > 150 and r < 100 and b < 100:
            return self.COLOR_NAMES_RU['green']
        elif r > 200 and g > 200 and b < 100:
            return self.COLOR_NAMES_RU['yellow']
        elif r > 200 and g > 100 and b < 100:
            return self.COLOR_NAMES_RU['orange']
        elif r > 150 and b > 150 and g < 100:
            return self.COLOR_NAMES_RU['purple']
        elif r > 200 and g > 150 and b > 150:
            return self.COLOR_NAMES_RU['pink']
        elif r > 100 and g > 50 and b < 50:
            return self.COLOR_NAMES_RU['brown']
        else:
            return 'unknown'
    
    def get_color_harmony_score(
        self,
        colors1: List[ColorInfo],
        colors2: List[ColorInfo]
    ) -> float:
        """
        Оценить цветовую гармонию между двумя наборами цветов
        
        Returns:
            Score от 0.0 до 1.0
        """
        if not colors1 or not colors2:
            return 0.5
        
        # Проверка на комплементарные цвета (противоположные на круге)
        score = 0.0
        
        primary1 = colors1[0].rgb
        primary2 = colors2[0].rgb
        
        # Конвертация в HSV
        hsv1 = self._rgb_to_hsv(primary1)
        hsv2 = self._rgb_to_hsv(primary2)
        
        # Разница в оттенке
        hue_diff = abs(hsv1[0] - hsv2[0])
        if hue_diff > 90:
            hue_diff = 180 - hue_diff
        
        # Комплементарные (~90° на круге) - хорошо
        if 70 <= hue_diff <= 110:
            score += 0.4
        
        # Аналогичные (~30°) - тоже хорошо
        if hue_diff <= 30:
            score += 0.3
        
        # Проверка нейтральных цветов
        neutrals = ['black', 'white', 'gray', 'beige', 'navy', 'brown']
        
        is_neutral1 = any(n in (colors1[0].color_name or '') for n in neutrals)
        is_neutral2 = any(n in (colors2[0].color_name or '') for n in neutrals)
        
        # Один нейтральный + один яркий = классическое сочетание
        if is_neutral1 != is_neutral2:  # XOR
            score += 0.3
        
        # Оба нейтральные = безопасно, но скучно
        if is_neutral1 and is_neutral2:
            score += 0.2
        
        return min(1.0, score + 0.3)  # Базовый score
    
    def _rgb_to_hsv(self, rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
        """Конвертировать RGB в HSV"""
        r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
        
        max_c = max(r, g, b)
        min_c = min(r, g, b)
        diff = max_c - min_c
        
        if diff == 0:
            h = 0
        elif max_c == r:
            h = (60 * ((g - b) / diff) + 360) % 360
        elif max_c == g:
            h = (60 * ((b - r) / diff) + 120) % 360
        else:
            h = (60 * ((r - g) / diff) + 240) % 360
        
        s = 0 if max_c == 0 else (diff / max_c) * 100
        v = max_c * 100
        
        return (h, s, v)
    
    def suggest_complementary_colors(
        self,
        base_color: ColorInfo
    ) -> List[str]:
        """
        Предложить цвета, которые гармоничны с базовым
        
        Returns:
            Список названий цветов на русском
        """
        rgb = base_color.rgb
        hsv = self._rgb_to_hsv(rgb)
        hue = hsv[0]
        
        suggestions = []
        
        # Комплементарный (противоположный)
        comp_hue = (hue + 180) % 360
        if 0 <= comp_hue < 30 or 330 <= comp_hue <= 360:
            suggestions.append(self.COLOR_NAMES_RU['red'])
        elif 30 <= comp_hue < 60:
            suggestions.append(self.COLOR_NAMES_RU['orange'])
        elif 60 <= comp_hue < 90:
            suggestions.append(self.COLOR_NAMES_RU['yellow'])
        elif 90 <= comp_hue < 150:
            suggestions.append(self.COLOR_NAMES_RU['green'])
        elif 150 <= comp_hue < 210:
            suggestions.append(self.COLOR_NAMES_RU['cyan'])
        elif 210 <= comp_hue < 270:
            suggestions.append(self.COLOR_NAMES_RU['blue'])
        elif 270 <= comp_hue < 330:
            suggestions.append(self.COLOR_NAMES_RU['purple'])
        
        # Всегда добавляем нейтральные
        suggestions.extend([self.COLOR_NAMES_RU['black'], self.COLOR_NAMES_RU['white'], self.COLOR_NAMES_RU['gray']])
        
        return list(set(suggestions))  # Уникальные


if __name__ == "__main__":
    # Тест
    analyzer = ColorAnalyzer()
    
    # Создаём тестовое изображение (красное)
    image = np.ones((200, 200, 3), dtype=np.uint8) * 50
    image[:, :] = [50, 50, 200]  # BGR - красный
    
    colors = analyzer.extract_dominant_colors(image)
    print("Доминирующие цвета:")
    for c in colors[:3]:
        print(f"  {c.color_name}: {c.hex} ({c.percentage}%)")
    
    # Тест гармонии
    blue_img = np.ones((100, 100, 3), dtype=np.uint8) * 200  # Синий
    orange_img = np.ones((100, 100, 3), dtype=np.uint8)
    orange_img[:, :] = [50, 150, 255]  # Оранжевый
    
    blue_colors = analyzer.extract_dominant_colors(blue_img)
    orange_colors = analyzer.extract_dominant_colors(orange_img)
    
    harmony = analyzer.get_color_harmony_score(blue_colors, orange_colors)
    print(f"\nГармония синий-оранжевый: {harmony:.2f}")
