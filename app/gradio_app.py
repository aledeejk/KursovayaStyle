import base64
import io
import os

import gradio as gr
import httpx
import numpy as np
from PIL import Image

API_URL = os.getenv("FASHION_API_URL", "http://127.0.0.1:8000")

ITEM_TYPE_CHOICES: list = [
    "— авто —",
    "t-shirt (футболка)",
    "shirt (рубашка)",
    "sweater (свитер)",
    "hoodie (худи)",
    "blouse (блузка)",
    "jacket (куртка/пиджак)",
    "blazer (пиджак)",
    "coat (пальто)",
    "jeans (джинсы)",
    "trousers (брюки)",
    "shorts (шорты)",
    "skirt (юбка)",
    "dress (платье)",
    "sneakers (кроссовки)",
    "boots (ботинки)",
    "sandals (сандали)",
    "loafers (лоферы)",
    "backpack (рюкзак)",
    "handbag (сумка)",
]


def _pil_to_bytes(pil_img: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def analyze_image(
    image: np.ndarray,
    style: str,
    item_type: str,
    randomize: bool,
    return_image: bool,
) -> tuple:
    """
    Called by Gradio on each submission.

    Returns:
        (annotated_pil_or_none, result_text)
    """
    if image is None:
        return None, "⚠️ Загрузите изображение."

    pil_img = Image.fromarray(image.astype("uint8"))
    img_bytes = _pil_to_bytes(pil_img)

    params: dict = {
        "style": style,
        "return_image": str(return_image).lower(),
        "randomize": str(randomize).lower(),
    }
    if item_type and item_type != "— авто —":
        params["item_type"] = item_type.split()[0]

    try:
        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{API_URL}/recommend",
                files={"file": ("image.jpg", img_bytes, "image/jpeg")},
                params=params,
            )
        resp.raise_for_status()
        data = resp.json()
    except httpx.ConnectError:
        return None, (
            "❌ Не удалось подключиться к API.\n"
            "Убедитесь, что сервер запущен:\n"
            "  python run.py"
        )
    except Exception as exc:
        return None, f"❌ Ошибка: {exc}"

    annotated_pil = None
    if data.get("annotated_image_b64"):
        raw = base64.b64decode(data["annotated_image_b64"])
        annotated_pil = Image.open(io.BytesIO(raw))

    lines = []

    auto_mode = not item_type or item_type == "— авто —"
    detected = data.get("detected_items", [])
    items_detail = data.get("items_detail", [])

    if detected and items_detail:
        lines.append(f"### 🔍 Автодетекция: обнаружено {len(items_detail)} предметов")
        for det_item in items_detail:
            lines.append(
                f"- **{det_item['class_name']}** — уверенность {det_item['confidence']:.0%}"
            )
    elif not auto_mode:
        lines.append(f"### 📌 Ручной выбор: **{item_type.split()[0]}**")
    else:
        lines.append("### ℹ️ Автодетекция не нашла одежду — использую базовые рекомендации")

    present = data.get("present_categories", [])
    missing = data.get("missing_categories", [])
    if present:
        lines.append(f"\n### 👗 Категории в образе")
        lines.append(", ".join(present))
    if missing:
        lines.append(f"\n### ➕ Не хватает для завершённого образа")
        lines.append(", ".join(missing))

    suggestions = data.get("suggestions", {})
    if suggestions:
        lines.append("\n### 💡 Что добавить")
        for cat, items in suggestions.items():
            lines.append(f"**{cat}:** {', '.join(items)}")

    advice = data.get("advice", "")
    if advice:
        lines.append(f"\n### 🎨 Совет по образу")
        lines.append(advice)

    color_tips = data.get("color_tips", [])
    if color_tips:
        lines.append("\n### 🎭 Цветовые сочетания")
        for tip in color_tips:
            lines.append(f"- {tip}")

    llm_advice = data.get("llm_advice", "")
    if llm_advice:
        lines.append("\n### ✨ Рекомендация стилиста")
        lines.append(llm_advice)

    result_text = "\n".join(lines)
    return annotated_pil, result_text


with gr.Blocks(title="Fashion AI – Outfit Recommender") as demo:
    gr.Markdown(
        """
        # 👗 Fashion AI — Распознавание одежды и генерация образов
        Загрузите фото, выберите стиль и нажмите **Анализировать**.
        Если автодетекция не сработала — выберите тип предмета вручную.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(
                label="Загрузите фото",
                type="numpy",
                sources=["upload", "clipboard"],
            )
            style_radio = gr.Radio(
                choices=["casual", "formal", "sporty"],
                value="casual",
                label="Желаемый стиль",
            )
            gr.Markdown(
                "💡 Загрузите фото — система автоматически распознаёт одежду через CLIP. Если не распознала — выберите тип вручную."
            )
            item_type_dropdown = gr.Dropdown(
                choices=ITEM_TYPE_CHOICES,
                value="— авто —",
                label="Тип предмета (если авто не сработало)",
            )
            with gr.Row():
                show_annotated = gr.Checkbox(
                    value=True,
                    label="Показать аннотацию",
                )
                randomize_checkbox = gr.Checkbox(
                    value=False,
                    label="🎲 Разнообразные советы",
                )
            submit_btn = gr.Button("🔍 Анализировать образ", variant="primary")

        with gr.Column(scale=1):
            output_image = gr.Image(
                label="Детекция одежды",
                type="pil",
                interactive=False,
            )
            output_text = gr.Markdown(label="Рекомендации")

    submit_btn.click(
        fn=analyze_image,
        inputs=[input_image, style_radio, item_type_dropdown, randomize_checkbox, show_annotated],
        outputs=[output_image, output_text],
    )

    gr.Markdown(
        "> Пример: выберите **sweater** → загрузите фото и получите рекомендации для свитера. Поставьте ✅ на «Разнообразные советы» — каждый раз будет другой набор."
    )

    gr.Examples(
        examples=[],
        inputs=input_image,
        label="Примеры (добавьте свои изображения)",
    )

    gr.Markdown(
        """
        ---
        **Стек:** YOLOv8 (детекция) · CLIP `openai/clip-vit-base-patch32` (эмбеддинги) · FastAPI · Gradio  
        Для LLM-совета задайте переменную `OPENAI_API_KEY`.
        """
    )


if __name__ == "__main__":
    import socket
    import sys

    _PORT = 7860
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as _s:
        _s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            _s.bind(("0.0.0.0", _PORT))
        except OSError:
            print(
                f"[ERROR] Port {_PORT} is busy.\n"
                f"Close the program using port {_PORT} and try again."
            )
            sys.exit(1)

    print(f"[*] Gradio UI: http://localhost:{_PORT}")
    demo.launch(
        server_name="0.0.0.0",
        server_port=_PORT,
        share=False,
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1100px !important; }",
    )
