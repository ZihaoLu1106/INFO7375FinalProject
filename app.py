import gradio as gr
from lavis.models import load_model_and_preprocess
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model, vis_processors, txt_processors = load_model_and_preprocess("blip2_caption", "pretrain", device=device)

def generate_caption(image):
    image = vis_processors["eval"](image).unsqueeze(0).to(device)
    caption = model.generate({"image": image})[0]
    return caption

gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="BLIP-2 Caption Generator (Flickr8k Fine-tuned)"
).launch()