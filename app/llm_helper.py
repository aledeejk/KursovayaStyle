import os
from typing import Dict, List, Optional


def generate_style_advice(
    detected_items: List[str],
    suggestions: Dict[str, List[str]],
    style: str = "casual",
    openai_api_key: Optional[str] = None,
) -> str:
    """
    Generate a natural-language styling advice string.

    If OPENAI_API_KEY is available (env var or parameter), calls GPT-3.5-turbo.
    Otherwise falls back to a deterministic rule-based template.

    Args:
        detected_items: raw class names from detector
        suggestions:    category -> list of suggested items
        style:          outfit style string
        openai_api_key: override env var

    Returns:
        Plain-text fashion advice paragraph.
    """
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

        items_str = ", ".join(detected_items) if detected_items else "ничего не обнаружено"
        sugg_lines = []
        for cat, items in suggestions.items():
            sugg_lines.append(f"  {cat}: {', '.join(items)}")
        sugg_str = "\n".join(sugg_lines) if sugg_lines else "нет рекомендаций"

        prompt = (
            f"Ты — стилист-эксперт. Пользователь носит: {items_str}.\n"
            f"Желаемый стиль: {style}.\n"
            f"Система предлагает добавить:\n{sugg_str}\n\n"
            "Напиши короткий (3–4 предложения) совет на русском языке: "
            "как дополнить образ, какие цвета и фасоны выбрать, чтобы получился стильный лук."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        return _rule_based_advice(detected_items, suggestions, style) + f" (OpenAI недоступен: {exc})"


def _rule_based_advice(
    detected_items: List[str],
    suggestions: Dict[str, List[str]],
    style: str,
) -> str:
    style_tips = {
        "casual": (
            "Для повседневного образа делайте ставку на базовые вещи: белая футболка, "
            "тёмные джинсы и белые кроссовки никогда не подводят."
        ),
        "formal": (
            "Для формального стиля выбирайте монохромные или нейтральные оттенки, "
            "аккуратный крой и минимум принтов."
        ),
        "sporty": (
            "Спортивный образ выигрывает от функциональных тканей, контрастных акцентов "
            "и удобной обуви для активности."
        ),
    }

    base = style_tips.get(style, style_tips["casual"])

    add_parts = []
    for cat, items in list(suggestions.items())[:2]:
        if items:
            add_parts.append(f"Хорошим дополнением станут {items[0]}")

    if add_parts:
        base += " " + ". ".join(add_parts) + "."

    return base
