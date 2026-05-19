import random
from typing import List, Dict, Optional, Set


NON_CLOTHING = {"person", "full_body", "other"}

CATEGORY_MAP: Dict[str, str] = {
    "person": "full_body",  # intentionally excluded downstream
    "shirt": "top", "t-shirt": "top", "top": "top", "blouse": "top",
    "sweater": "top", "hoodie": "top", "vest": "top",
    "short_sleeved_shirt": "top", "long_sleeved_shirt": "top",
    "jacket": "outerwear", "coat": "outerwear", "blazer": "outerwear",
    "short_sleeved_outwear": "outerwear", "long_sleeved_outwear": "outerwear",
    "suit": "outerwear",
    "dress": "dress", "short_sleeved_dress": "dress",
    "long_sleeved_dress": "dress", "vest_dress": "dress", "sling_dress": "dress",
    "skirt": "bottom", "pants": "bottom", "jeans": "bottom",
    "shorts": "bottom", "leggings": "bottom", "trousers": "bottom",
    "overalls": "bottom", "jumpsuit": "bottom", "sling": "bottom",
    "bottom": "bottom",
    "shoes": "shoes", "shoe": "shoes", "boot": "shoes", "sneaker": "shoes",
    "sandal": "shoes", "heel": "shoes",
    "hat": "accessory", "cap": "accessory", "scarf": "accessory",
    "belt": "accessory", "glove": "accessory", "tie": "accessory",
    "handbag": "bag", "backpack": "bag", "bag": "bag",
    "umbrella": "accessory", "suitcase": "bag",
}

OUTFIT_RULES: Dict[str, List[List[str]]] = {
    "casual": [
        ["top", "bottom", "shoes"],
        ["top", "bottom", "shoes", "bag"],
        ["top", "bottom", "outerwear", "shoes"],
        ["dress", "shoes"],
        ["dress", "shoes", "bag"],
    ],
    "formal": [
        ["outerwear", "bottom", "shoes"],
        ["dress", "shoes", "bag"],
        ["top", "outerwear", "bottom", "shoes"],
    ],
    "sporty": [
        ["top", "bottom", "shoes"],
        ["top", "bottom", "shoes", "bag"],
    ],
}

COMPLEMENT_SUGGESTIONS: Dict[str, Dict[str, List[str]]] = {
    "top": {
        "bottom": ["классические джинсы", "брюки чинос", "юбка-миди", "прямые брюки"],
        "shoes": ["белые кроссовки", "лоферы", "ботинки дерби"],
        "outerwear": ["джинсовая куртка", "лёгкий бомбер", "тренч"],
        "bag": ["холщовая сумка-тоут", "небольшой рюкзак"],
    },
    "bottom": {
        "top": ["базовая белая футболка", "полосатый топ", "тонкий джемпер"],
        "shoes": ["белые сникеры", "классические туфли", "ботильоны"],
        "outerwear": ["пиджак оверсайз", "лёгкая ветровка"],
        "bag": ["кросс-боди", "клатч"],
    },
    "dress": {
        "shoes": ["босоножки на каблуке", "белые кеды", "ботильоны"],
        "outerwear": ["джинсовая куртка", "пальто"],
        "bag": ["маленькая сумочка", "клатч"],
        "accessory": ["поясной ремень", "шёлковый платок"],
    },
    "outerwear": {
        "top": ["базовая водолазка", "тонкая рубашка"],
        "bottom": ["прямые джинсы", "тёмные брюки"],
        "shoes": ["кожаные ботинки", "лоферы"],
        "bag": ["кожаная сумка"],
    },
    "shoes": {
        "top": ["лёгкая блузка", "базовый джемпер"],
        "bottom": ["подвёрнутые джинсы", "летние брюки"],
        "bag": ["маленькая сумка через плечо"],
    },
    "bag": {
        "top": ["шёлковая блузка", "простой трикотажный топ"],
        "shoes": ["туфли в тон", "кроссовки"],
    },
    "accessory": {
        "top": ["однотонный топ без принта"],
        "bottom": ["однотонные брюки"],
    },
}

SPECIFIC_RECOMMENDATIONS: Dict[str, Dict[str, List[str]]] = {
    "t-shirt": {
        "bottom": ["классические джинсы", "шорты", "спортивные штаны"],
        "shoes": ["белые кеды", "сникеры", "сандалии"],
        "outerwear": ["джинсовая куртка", "бомбер", "худи"],
        "bag": ["рюкзак", "поясная сумка"],
    },
    "shirt": {
        "bottom": ["брюки чинос", "прямые джинсы", "юбка-миди"],
        "shoes": ["лоферы", "ботинки дерби", "сникеры"],
        "outerwear": ["пиджак оверсайз", "тренч"],
        "bag": ["сумка-тоут", "кожаная сумка"],
        "accessory": ["часы", "тонкий ремень"],
    },
    "long_sleeved_shirt": {
        "bottom": ["прямые джинсы", "брюки чинос"],
        "shoes": ["лоферы", "ботинки дерби"],
        "outerwear": ["пиджак", "тренч"],
        "bag": ["сумка-тоут"],
    },
    "sweater": {
        "bottom": ["прямые джинсы", "шерстяные брюки", "юбка плиссе"],
        "shoes": ["ботинки чанки", "лоферы", "сникеры"],
        "outerwear": ["длинное пальто", "тренч"],
        "bag": ["кожаная сумка", "корзина"],
        "accessory": ["шерстяной шарф", "базовая шапка"],
    },
    "hoodie": {
        "bottom": ["джоггеры", "джинсы скинни", "спортивные шорты"],
        "shoes": ["кроссовки", "сникеры", "слипоны"],
        "outerwear": ["бомбер", "ветровка"],
        "bag": ["рюкзак"],
    },
    "blouse": {
        "bottom": ["брюки палаццо", "юбка-карандаш", "прямые джинсы"],
        "shoes": ["балетки", "лоферы", "босоножки на каблуке"],
        "bag": ["клатч", "маленькая сумочка"],
        "accessory": ["тонкое ожерелье", "шёлковый платок"],
    },
    "vest": {
        "bottom": ["джинсы скинни", "брюки чинос"],
        "shoes": ["белые сникеры", "лоферы"],
        "outerwear": ["джинсовая куртка"],
        "bag": ["рюкзак"],
    },
    "jacket": {
        "top": ["белая футболка", "полосатая рубашка", "тонкий джемпер"],
        "bottom": ["прямые джинсы", "брюки чинос"],
        "shoes": ["кожаные ботинки", "лоферы", "сникеры"],
        "bag": ["кожаный рюкзак", "мессенджер"],
    },
    "coat": {
        "top": ["базовая водолазка", "тонкий свитер", "рубашка"],
        "bottom": ["прямые брюки", "юбка-миди", "джинсы"],
        "shoes": ["кожаные ботинки", "ботильоны"],
        "accessory": ["шерстяной шарф", "кожаные перчатки", "шапка-бини"],
        "bag": ["кожаная сумка"],
    },
    "blazer": {
        "top": ["базовая белая футболка", "шёлковая блузка"],
        "bottom": ["брюки дудочки", "джинсы прямого кроя"],
        "shoes": ["туфли-лодочки", "лоферы", "ботинки дерби"],
        "bag": ["структурированная сумка", "клатч"],
        "accessory": ["часы", "тонкое ожерелье"],
    },
    "suit": {
        "top": ["рубашка с воротником", "водолазка"],
        "shoes": ["туфли дерби", "монки", "оксфорды"],
        "accessory": ["галстук или платок в кармане", "часы"],
        "bag": ["портфель", "кожаная папка"],
    },
    "dress": {
        "shoes": ["босоножки на каблуке", "белые кеды", "ботильоны", "балетки"],
        "outerwear": ["джинсовая куртка", "тренч", "лёгкое пальто"],
        "bag": ["клатч", "маленькая сумочка", "плетёная сумка"],
        "accessory": ["поясной ремень", "шёлковый платок", "тонкое украшение"],
    },
    "jeans": {
        "top": ["белая футболка", "джемпер оверсайз", "полосатая рубашка"],
        "shoes": ["белые сникеры", "ботинки", "лоферы"],
        "outerwear": ["бомбер", "джинсовая куртка", "кожаная куртка"],
        "bag": ["рюкзак", "кросс-боди", "холщовая сумка"],
    },
    "pants": {
        "top": ["базовая белая футболка", "трикотажный топ", "рубашка"],
        "shoes": ["сникеры", "лоферы", "ботинки"],
        "outerwear": ["пиджак", "ветровка"],
        "bag": ["кросс-боди", "клатч"],
    },
    "trousers": {
        "top": ["шёлковая блузка", "водолазка", "строгая рубашка"],
        "shoes": ["туфли-лодочки", "лоферы", "ботильоны"],
        "outerwear": ["пиджак", "пальто"],
        "bag": ["структурированная сумка", "клатч"],
    },
    "shorts": {
        "top": ["белая футболка", "топ с принтом", "рубашка навыпуск"],
        "shoes": ["сандалии", "сникеры", "слипоны"],
        "bag": ["маленький рюкзак", "пляжная сумка"],
    },
    "skirt": {
        "top": ["базовый топ", "трикотажная майка", "рубашка"],
        "shoes": ["босоножки", "балетки", "белые кеды"],
        "outerwear": ["оверсайз-пиджак", "джинсовая куртка"],
        "bag": ["кросс-боди", "мини-сумка"],
    },
    "leggings": {
        "top": ["длинная туника", "оверсайз-свитер", "спортивный лонгслив"],
        "shoes": ["кроссовки", "угги", "балетки"],
        "outerwear": ["спортивная куртка", "дутое пальто"],
        "bag": ["рюкзак"],
    },
    "boots": {
        "bottom": ["прямые джинсы", "юбка-миди", "кожаные брюки"],
        "top": ["тонкий свитер", "водолазка"],
        "outerwear": ["пальто", "дублёнка"],
    },
    "sneakers": {
        "bottom": ["джинсы", "джоггеры", "шорты"],
        "top": ["футболка", "худи", "спортивный лонгслив"],
        "bag": ["рюкзак", "поясная сумка"],
    },
    "sandals": {
        "bottom": ["лёгкие брюки", "юбка", "шорты"],
        "top": ["лёгкая блузка", "майка", "топ"],
        "bag": ["соломенная сумка", "клатч"],
    },
    "backpack": {
        "top": ["худи", "футболка"],
        "bottom": ["джинсы", "спортивные штаны"],
        "shoes": ["кроссовки", "сникеры"],
    },
    "handbag": {
        "top": ["блузка", "трикотажный топ"],
        "shoes": ["туфли", "лоферы"],
    },
}

TYPE_RECOMMENDATIONS: Dict[str, Dict[str, List[str]]] = {
    "t-shirt":   {"bottom": ["джинсы", "шорты"],              "shoes": ["кеды", "сникеры"]},
    "shirt":     {"bottom": ["брюки чинос", "юбка"],          "shoes": ["лоферы", "дерби"]},
    "sweater":   {"bottom": ["джинсы", "брюки"],              "shoes": ["ботинки", "челси"]},
    "hoodie":    {"bottom": ["спортивные штаны", "джинсы"],   "shoes": ["кроссовки"]},
    "blouse":    {"bottom": ["брюки палаццо", "юбка-карандаш"], "shoes": ["балетки", "лоферы"]},
    "jacket":    {"bottom": ["джинсы", "брюки"],  "shoes": ["ботинки", "кеды"],    "top": ["свитер", "футболка"]},
    "blazer":    {"bottom": ["брюки", "джинсы"], "shoes": ["лоферы", "туфли"],     "top": ["рубашка", "футболка"]},
    "coat":      {"bottom": ["брюки", "юбка"],    "shoes": ["сапоги", "ботинки"],  "accessory": ["шарф"]},
    "jeans":     {"top": ["футболка", "свитер"],              "shoes": ["кеды", "ботинки"]},
    "trousers":  {"top": ["рубашка", "свитер"],               "shoes": ["лоферы", "туфли"]},
    "shorts":    {"top": ["футболка", "топ"],                 "shoes": ["сникеры", "сандалии"]},
    "skirt":     {"top": ["топ", "блузка"],                   "shoes": ["балетки", "ботильоны"]},
    "dress":     {"shoes": ["лодочки", "балетки"],            "bag": ["клатч"]},
    "sneakers":  {"bottom": ["джинсы", "шорты"],              "top": ["футболка", "худи"]},
    "boots":     {"bottom": ["джинсы", "брюки"],              "top": ["свитер", "куртка"]},
    "sandals":   {"bottom": ["юбка", "шорты"],                "top": ["топ", "лёгкая блузка"]},
    "backpack":  {"top": ["худи", "футболка"],                "bottom": ["джинсы", "спортивные штаны"]},
    "handbag":   {"top": ["блузка", "трикотажный топ"],       "shoes": ["туфли", "лоферы"]},
    "hat":       {"top": ["футболка", "худи"],                "bottom": ["джинсы"]},
    "scarf":     {"outerwear": ["пальто", "куртка"],          "bottom": ["брюки", "джинсы"]},
}

COLOR_NEUTRALS = {"black", "white", "grey", "gray", "beige", "cream", "navy", "tan", "brown"}

COLOR_COMPLEMENTS: Dict[str, List[str]] = {
    "red": ["белый", "чёрный", "серый", "джинсово-синий"],
    "blue": ["белый", "бежевый", "коричневый", "горчичный"],
    "green": ["белый", "бежевый", "коричневый", "чёрный"],
    "yellow": ["белый", "серый", "чёрный", "синий денима"],
    "pink": ["белый", "серый", "бежевый", "светло-синий"],
    "orange": ["белый", "синий денима", "чёрный", "коричневый"],
    "purple": ["серый", "бежевый", "белый", "чёрный"],
    "brown": ["бежевый", "белый", "кремовый", "горчичный"],
    "black": ["белый", "серый", "красный", "любой пастельный"],
    "white": ["чёрный", "синий денима", "любой насыщенный цвет"],
    "grey": ["белый", "чёрный", "горчичный", "бургунди"],
    "beige": ["коричневый", "белый", "чёрный", "оливковый"],
    "navy": ["белый", "бежевый", "красный", "полоска"],
}


def _build_specific_suggestions(
    detected_names: List[str],
    missing: List[str],
    present_categories: List[str],
    randomize: bool = False,
) -> Dict[str, List[str]]:
    """
    Build suggestions with three-level priority:
      1. TYPE_RECOMMENDATIONS   — short, targeted Russian phrases (highest priority)
      2. SPECIFIC_RECOMMENDATIONS — detailed Russian phrases (fills gaps)
      3. COMPLEMENT_SUGGESTIONS — generic category-level fallback

    Results are deduplicated and capped at 4 items per category.
    When randomize=True items are shuffled before capping.
    """
    suggestions: Dict[str, List[str]] = {}

    def _add(cat: str, items: List[str]) -> None:
        if cat in missing or cat not in present_categories:
            suggestions.setdefault(cat, []).extend(items)

    for name in detected_names:
        key = name.lower()
        if key in TYPE_RECOMMENDATIONS:
            for dep_cat, dep_items in TYPE_RECOMMENDATIONS[key].items():
                _add(dep_cat, dep_items)
        if key in SPECIFIC_RECOMMENDATIONS:
            for dep_cat, dep_items in SPECIFIC_RECOMMENDATIONS[key].items():
                _add(dep_cat, dep_items)

    for cat in present_categories:
        if cat in COMPLEMENT_SUGGESTIONS:
            for dep_cat, dep_items in COMPLEMENT_SUGGESTIONS[cat].items():
                _add(dep_cat, dep_items)

    for key in suggestions:
        unique = list(dict.fromkeys(suggestions[key]))
        if randomize:
            random.shuffle(unique)
        suggestions[key] = unique[:4]

    return suggestions


def classify_detected_items(detected_names: List[str]) -> Dict[str, List[str]]:
    """
    Groups raw class names into fashion categories, excluding non-clothing entries.

    Returns:
        dict mapping category -> list of raw names (never includes full_body/person/other)
    """
    grouped: Dict[str, List[str]] = {}
    for name in detected_names:
        cat = CATEGORY_MAP.get(name.lower(), "other")
        if cat in NON_CLOTHING:
            continue
        grouped.setdefault(cat, []).append(name)
    return grouped


def _default_missing(style: str) -> List[str]:
    """Return the canonical full-outfit template for a style (used when nothing detected)."""
    templates = OUTFIT_RULES.get(style, OUTFIT_RULES["casual"])
    return list(templates[0])


def _suggestions_for_empty(
    missing_cats: List[str],
    manual_item_name: Optional[str] = None,
    randomize: bool = False,
) -> Dict[str, List[str]]:
    """
    Build suggestion dict when nothing was auto-detected.
    If manual_item_name is an exact key in SPECIFIC_RECOMMENDATIONS, use those first.
    """
    present: List[str] = []
    suggestions: Dict[str, List[str]] = {}

    if manual_item_name:
        key = manual_item_name.lower()
        if key in SPECIFIC_RECOMMENDATIONS:
            for dep_cat, dep_items in SPECIFIC_RECOMMENDATIONS[key].items():
                if dep_cat in missing_cats:
                    suggestions.setdefault(dep_cat, []).extend(dep_items)

    for cat in missing_cats:
        if cat in COMPLEMENT_SUGGESTIONS:
            for dep_cat, dep_items in COMPLEMENT_SUGGESTIONS[cat].items():
                if dep_cat in missing_cats:
                    suggestions.setdefault(dep_cat, []).extend(dep_items)

    for key in suggestions:
        unique = list(dict.fromkeys(suggestions[key]))
        if randomize:
            random.shuffle(unique)
        suggestions[key] = unique[:4]

    return suggestions


def generate_outfit(
    detected_items: List[str],
    style: str = "casual",
    detected_colors: Optional[List[str]] = None,
    manual_item_type: Optional[str] = None,
    randomize: bool = False,
) -> Dict:
    """
    Generate outfit recommendation based on detected items.

    Args:
        detected_items:   raw class names from detector
        style:            one of 'casual', 'formal', 'sporty'
        detected_colors:  optional dominant colour names
        manual_item_type: user-supplied category override (e.g. 'top', 'bottom', 'shoes')
                          Added to present_categories when automatic detection is unreliable.

    Returns:
        dict with keys: detected, present, missing, suggestions, advice, color_tips
    """
    grouped = classify_detected_items(detected_items)

    if manual_item_type:
        cat = CATEGORY_MAP.get(manual_item_type.lower(), manual_item_type.lower())
        if cat not in NON_CLOTHING:
            grouped.setdefault(cat, [manual_item_type])

    present_categories = list(grouped.keys())

    if not present_categories:
        missing = _default_missing(style)
        suggestions = _suggestions_for_empty(missing, manual_item_name=manual_item_type, randomize=randomize)
        return {
            "detected": detected_items,
            "present": [],
            "missing": missing,
            "suggestions": suggestions,
            "advice": _build_advice([], missing, style),
            "color_tips": [],
        }

    rules = OUTFIT_RULES.get(style, OUTFIT_RULES["casual"])

    best_template: Optional[List[str]] = None
    best_overlap = -1
    for template in rules:
        overlap = sum(1 for cat in template if cat in present_categories)
        if overlap > best_overlap:
            best_overlap = overlap
            best_template = template

    missing: List[str] = []
    if best_template:
        missing = [cat for cat in best_template if cat not in present_categories]

    all_detected_names: List[str] = []
    for names in grouped.values():
        all_detected_names.extend(names)

    suggestions = _build_specific_suggestions(all_detected_names, missing, present_categories, randomize=randomize)

    color_tips: List[str] = []
    if detected_colors:
        for color in detected_colors:
            color_lower = color.lower()
            if color_lower in COLOR_COMPLEMENTS:
                tips = COLOR_COMPLEMENTS[color_lower]
                color_tips.append(
                    f"К {color_lower} подойдут: {', '.join(tips[:3])}."
                )

    advice = _build_advice(present_categories, missing, style)

    return {
        "detected": detected_items,
        "present": present_categories,
        "missing": missing,
        "suggestions": suggestions,
        "advice": advice,
        "color_tips": color_tips,
    }


def _build_advice(present: List[str], missing: List[str], style: str) -> str:
    lines = []
    if not missing:
        lines.append(f"Образ в стиле «{style}» выглядит завершённым!")
    else:
        readable_missing = ", ".join(_ru_category(c) for c in missing)
        lines.append(f"Для завершённого образа «{style}» добавьте: {readable_missing}.")

    if "dress" in present:
        lines.append("С платьем — минимум лишних деталей: туфли или кеды плюс одна сумочка.")
    if "outerwear" in present and "bottom" in present:
        lines.append("Верхняя одежда + брюки — классика: выберите базовую рубашку или водолазку внутрь.")
    if "top" in present and "bottom" in present:
        lines.append("Верх и низ есть — убедитесь, что цвета сочетаются (нейтральный + акцентный).")

    return " ".join(lines)


def _ru_category(cat: str) -> str:
    mapping = {
        "top": "верх",
        "bottom": "низ",
        "outerwear": "верхняя одежда",
        "dress": "платье",
        "shoes": "обувь",
        "bag": "сумка",
        "accessory": "аксессуар",
    }
    return mapping.get(cat, cat)
