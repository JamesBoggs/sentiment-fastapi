from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch
import os

MODEL_ID = os.getenv("MODEL_ID", "your-org/sentiment-large")
LORA_PATH = os.getenv("LORA_PATH", "./app/lora_adapter")

print(f"🔹 Loading base model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(LORA_PATH or MODEL_ID)

base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)

print(f"🔹 Attaching LoRA adapters from {LORA_PATH}")
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()

def get_model():
    return model, tokenizer
