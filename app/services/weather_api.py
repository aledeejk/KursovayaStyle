"""
Weather Integration Module
Интеграция с OpenWeatherMap API для учёта погоды в рекомендациях
"""
import aiohttp
import os
from typing import Optional, Dict
from dataclasses import dataclass
from enum import Enum


class WeatherCondition(Enum):
    """Условия погоды"""
    SUNNY = "sunny"          # Ясно, солнечно
    CLOUDY = "cloudy"        # Облачно
    RAINY = "rainy"          # Дождь
    SNOWY = "snowy"        # Снег
    HOT = "hot"              # Жарко (>25°C)
    COLD = "cold"            # Холодно (<10°C)
    WINDY = "windy"          # Ветрено


@dataclass
class WeatherInfo:
    """Информация о погоде"""
    temperature: float          # °C
    feels_like: float           # Ощущается как
    humidity: int               # %
    condition: WeatherCondition
    description: str            # Текстовое описание
    wind_speed: float           # м/с
    city: str
    
    def is_warm(self) -> bool:
        """Тепло ли?"""
        return self.temperature > 20
    
    def is_cold(self) -> bool:
        """Холодно ли?"""
        return self.temperature < 10
    
    def is_rainy(self) -> bool:
        """Дождь?"""
        return self.condition == WeatherCondition.RAINY
    
    def needs_coat(self) -> bool:
        """Нужна ли верхняя одежда?"""
        return self.temperature < 15 or self.condition in [WeatherCondition.RAINY, WeatherCondition.SNOWY]


class WeatherService:
    """
    Сервис для получения погоды
    
    Использует OpenWeatherMap API (бесплатный tier: 1000 calls/day)
    Получить API key: https://openweathermap.org/api
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Инициализация
        
        Args:
            api_key: Ключ API OpenWeatherMap. Если None, берётся из env WEATHER_API_KEY
        """
        self.api_key = api_key or os.getenv('WEATHER_API_KEY')
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    async def get_weather(
        self,
        city: str,
        country_code: Optional[str] = None
    ) -> Optional[WeatherInfo]:
        """
        Получить погоду для города
        
        Args:
            city: Название города (например, "Moscow", "London")
            country_code: Код страны ISO 3166 (например, "RU", "GB")
            
        Returns:
            WeatherInfo или None если ошибка
        """
        if not self.api_key:
            print("⚠️ WEATHER_API_KEY не установлен. Погода не будет учитываться.")
            return None
        
        # Формируем запрос
        q = f"{city},{country_code}" if country_code else city
        params = {
            'q': q,
            'appid': self.api_key,
            'units': 'metric',  # Цельсий
            'lang': 'ru'      # Русское описание
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_weather(data)
                    else:
                        print(f"Ошибка получения погоды: {response.status}")
                        return None
        except Exception as e:
            print(f"Ошибка запроса погоды: {e}")
            return None
    
    def _parse_weather(self, data: Dict) -> WeatherInfo:
        """Парсинг ответа API"""
        main = data['main']
        weather = data['weather'][0]
        wind = data.get('wind', {})
        
        temp = main['temp']
        
        # Определяем условие
        condition = self._map_condition(weather['id'], temp)
        
        return WeatherInfo(
            temperature=temp,
            feels_like=main['feels_like'],
            humidity=main['humidity'],
            condition=condition,
            description=weather['description'],
            wind_speed=wind.get('speed', 0),
            city=data['name']
        )
    
    def _map_condition(self, weather_id: int, temp: float) -> WeatherCondition:
        """
        Маппинг ID погоды OpenWeatherMap на наши категории
        
        Weather IDs:
        200-232: Thunderstorm
        300-321: Drizzle
        500-531: Rain
        600-622: Snow
        701-781: Atmosphere (fog, mist, etc.)
        800: Clear
        801-804: Clouds
        """
        # Сначала проверяем температуру
        if temp > 25:
            return WeatherCondition.HOT
        elif temp < 5:
            return WeatherCondition.COLD
        
        # Затем погодные условия
        if 200 <= weather_id <= 232:
            return WeatherCondition.RAINY  # Гроза
        elif 300 <= weather_id <= 321:
            return WeatherCondition.RAINY  # Морось
        elif 500 <= weather_id <= 531:
            return WeatherCondition.RAINY  # Дождь
        elif 600 <= weather_id <= 622:
            return WeatherCondition.SNOWY  # Снег
        elif weather_id == 800:
            return WeatherCondition.SUNNY  # Ясно
        elif 801 <= weather_id <= 804:
            return WeatherCondition.CLOUDY  # Облачно
        
        return WeatherCondition.CLOUDY
    
    def get_clothing_recommendations(
        self,
        weather: WeatherInfo
    ) -> Dict[str, list]:
        """
        Получить рекомендации по одежде на основе погоды
        
        Returns:
            Dict с категориями одежды
        """
        recommendations = {
            'must_have': [],  # Обязательно
            'recommended': [],  # Рекомендуется
            'avoid': [],  # Избегать
        }
        
        # Температура
        if weather.is_cold():
            recommendations['must_have'].extend(['куртка', 'свитер', 'шарф', 'перчатки'])
            recommendations['recommended'].extend(['водолазка', 'теплые брюки', 'ботинки'])
            recommendations['avoid'].extend(['футболка', 'шорты', 'юбка', 'сандалии'])
        
        elif weather.temperature < 15:
            recommendations['must_have'].append('лёгкая куртка')
            recommendations['recommended'].extend(['лонгслив', 'джинсы', 'кроссовки'])
        
        elif weather.temperature < 25:
            recommendations['recommended'].extend(['футболка', 'рубашка', 'джинсы', 'брюки'])
        
        else:  # Жарко
            recommendations['must_have'].extend(['лёгкая одежда', 'головной убор'])
            recommendations['recommended'].extend(['футболка', 'шорты', 'юбка', 'сандалии'])
            recommendations['avoid'].extend(['куртка', 'свитер', 'тёплые вещи'])
        
        # Осадки
        if weather.is_rainy():
            recommendations['must_have'].extend(['дождевик', 'зонт', 'непромокаемая обувь'])
            recommendations['avoid'].extend(['замша', 'кожа (необработанная)', 'светлая обувь'])
        
        if weather.condition == WeatherCondition.SNOWY:
            recommendations['must_have'].extend(['зимняя куртка', 'непромокаемые ботинки', 'шапка'])
        
        # Ветер
        if weather.wind_speed > 10:
            recommendations['recommended'].append('ветровка')
        
        return recommendations


# === ПРАВИЛА СЕЗОННОСТИ ===

SEASON_RULES = {
    'spring': {
        'recommended': ['тренч', 'лёгкая куртка', 'джинсы', 'кроссовки', 'лонгслив'],
        'colors': ['пастельные', 'бежевый', 'голубой', 'зелёный'],
    },
    'summer': {
        'recommended': ['футболка', 'шорты', 'юбка', 'платье', 'сандалии'],
        'colors': ['яркие', 'белый', 'синий', 'жёлтый'],
    },
    'fall': {
        'recommended': ['свитер', 'кардиган', 'пальто', 'ботинки', 'джинсы'],
        'colors': ['бордовый', 'горчичный', 'коричневый', 'оливковый'],
    },
    'winter': {
        'recommended': ['пуховик', 'свитер', 'водолазка', 'ботинки', 'шарф'],
        'colors': ['тёмные', 'чёрный', 'серый', 'синий'],
    },
}


def get_season_recommendations(season: str) -> Dict:
    """Получить рекомендации для сезона"""
    return SEASON_RULES.get(season.lower(), SEASON_RULES['spring'])


# === ПРИМЕР ИСПОЛЬЗОВАНИЯ ===

"""
# Получение погоды и рекомендаций:

import asyncio

async def main():
    weather_service = WeatherService(api_key="your_api_key")
    
    # Получаем погоду для Москвы
    weather = await weather_service.get_weather("Moscow", "RU")
    
    if weather:
        print(f"Погода в {weather.city}: {weather.temperature}°C, {weather.description}")
        
        # Получаем рекомендации
        recs = weather_service.get_clothing_recommendations(weather)
        print(f"Обязательно: {', '.join(recs['must_have'])}")
        print(f"Рекомендуется: {', '.join(recs['recommended'])}")
        print(f"Избегать: {', '.join(recs['avoid'])}")

asyncio.run(main())
"""

if __name__ == "__main__":
    # Тест без API ключа
    print("WeatherService инициализирован")
    print("Для использования установите WEATHER_API_KEY")
    print("Получить ключ: https://openweathermap.org/api")
