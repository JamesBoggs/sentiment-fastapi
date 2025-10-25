# app/model_loader.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import torch

MODEL_ID = "your-org/sentiment-large"
LORA_PATH = "./app/lora_adapter/"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base_model, LORA_PATH)
model.eval()
