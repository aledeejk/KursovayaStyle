import os
from typing import Dict, List, Optional

def generate_style_advice(
    detected_items: List[str],
    suggestions: Dict[str, List[str]],
    style: str = "casual",
    openai_api_key: Optional[str] = None,
) -> str:
    api_key = openai_api_key or os.getenv("OPENAI_API_KEY")

    if api_key:
        return _call_openai(detected_items, suggestions, style, api_key)

    return _rule_based_advice(detected_items, suggestions, style)


def _call_openai(
    detected_items: List[str],
    suggestions: Dict[str, List[str]],
    style: str,
    api_key: str,
) -> str:
    try:
        import openai

        client = openai.OpenAI(api_key=api_key)

        items_str = ", ".join(detected_items) if detected_items else "ничего"
        
        sugg_lines = []
        for cat, items in suggestions.items():
            if items:
                sugg_lines.append(f"{cat}: {', '.join(items[:3])}")
        
        prompt = f"""Ты стилист. Человек надел: {items_str}. Хочет выглядеть {style}.
Рекомендую добавить: {', '.join(sugg_lines) if sugg_lines else 'ничего конкретного'}.
Дай короткий совет (2-3 предложения) по-русски: что ещё можно надеть, как сочетать цвета и фасоны."""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return _rule_based_advice(detected_items, suggestions, style)


def _rule_based_advice(
    detected_items: List[str],
    suggestions: Dict[str, List[str]],
    style: str,
) -> str:
    tips = {
        "casual": "Берите базу: белую футболку, джинсы и кроссовки. Не прогадаете.",
        "formal": "Нейтральные тона, чёткий крой, минимум деталей. Классика.",
        "sporty": "Спорт: удобная обувь, свободный крой, контрастные цвета.",
    }
    
    result = tips.get(style, tips["casual"])
    
    for cat, items in suggestions.items():
        if items:
            result += f" Из {cat} добавьте {items[0]}."
            break
    
    return result