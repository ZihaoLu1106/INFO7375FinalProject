# scripts/finetune_blip2.py
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b").to(device)
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

from PIL import Image
import json

with open("data/flickr8k.json", "r") as f:
    dataset = json.load(f)

class Flickr8kDataset(Dataset):
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image = Image.open(entry["image"]).convert("RGB")
        caption = entry["caption"][0]

        inputs = self.processor(images=image, text=caption, return_tensors="pt", padding="max_length", truncation=True, max_length=64)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs

train_dataset = Flickr8kDataset(raw_data, processor)

training_args = TrainingArguments(
    output_dir="./blip2-flickr8k",
    per_device_train_batch_size=2,  # Reduce if you hit CUDA OOM
    num_train_epochs=5,
    fp16=True,
    save_steps=500,
    logging_steps=100,
    remove_unused_columns=False,
    report_to="none",
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=processor,
)

trainer.train()