import base64
import io
import os
import gradio as gr
import httpx
import numpy as np
from PIL import Image

API_URL = os.getenv("FASHION_API_URL", "http://127.0.0.1:8000")

ITEM_TYPES = [
    "t-shirt", "shirt", "sweater", "hoodie", "blouse", "jacket", "blazer", "coat",
    "jeans", "trousers", "shorts", "skirt", "dress", "sneakers", "boots", "sandals"
]

def analyze(image, style, manual_types, show_tips, return_annotated, gender):
    if image is None:
        return None, "Нет изображения"

    buf = io.BytesIO()
    Image.fromarray(image.astype("uint8")).save(buf, format="JPEG", quality=90)

    params = {"style": style, "return_image": str(return_annotated).lower(), "gender": gender}
    if manual_types:
        params["item_type"] = manual_types[0].split()[0]
    if show_tips:
        params["randomize"] = "true"

    try:
        resp = httpx.post(
            f"{API_URL}/recommend",
            files={"file": ("image.jpg", buf.getvalue(), "image/jpeg")},
            params=params,
            timeout=120
        )
        resp.raise_for_status()
        data = resp.json()
    except:
        return None, "API не отвечает"

    annotated = None
    if data.get("annotated_image_b64"):
        annotated = Image.open(io.BytesIO(base64.b64decode(data["annotated_image_b64"])))

        lines = []
    if data.get("items_detail"):
        items = data["items_detail"]
        lines.append(f"Найдено: {', '.join(i['class_name'] for i in items)}")
    elif manual_types:
        lines.append(f"Тип: {', '.join(manual_types)}")
    else:
        lines.append("Автодетекция не сработала")

    if data.get("present_categories"):
        lines.append(f"Категории: {', '.join(data['present_categories'])}")
    if data.get("missing_categories"):
        lines.append(f"Добавьте: {', '.join(data['missing_categories'])}")

    if show_tips and data.get("suggestions"):
        lines.append("\nВарианты:")
        for cat, items in data["suggestions"].items():
            lines.append(f"  {cat}: {', '.join(items[:2])}")

    return annotated, "\n\n".join(lines) 

with gr.Blocks(title="Fashion AI") as demo:
    gr.Markdown("# Fashion AI")

    with gr.Row():
        with gr.Column():
            img_in = gr.Image(label="Фото", type="numpy")
            style = gr.Radio(["casual", "formal", "sporty"], label="Стиль", value="casual")
            gender = gr.Radio(["auto", "male", "female"], label="Пол", value="auto")
            manual = gr.CheckboxGroup(choices=ITEM_TYPES, label="Тип одежды (если авто не сработал)")
            tips = gr.Checkbox(label="Советы")
            annot = gr.Checkbox(label="Аннотация", value=True)
            btn = gr.Button("Анализировать", variant="primary")

        with gr.Column():
            img_out = gr.Image(label="Результат")
            text_out = gr.Markdown(label="Рекомендации")

    btn.click(analyze, [img_in, style, manual, tips, annot, gender], [img_out, text_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", port=7860)