# 1. Imports
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


# 2. Initialize FastAPI app
app = FastAPI()


# 3. Load model & tokenizer
MODEL_NAME = "distilbert-base-uncased"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.eval()


# 4. Define request schema (USED BY ENDPOINTS)
class TextRequest(BaseModel):
    text: str


# 5. Health check endpoint
@app.get("/")
def health_check():
    return {"status": "API is running"}


# 6. Prediction endpoint
@app.post("/predict")
def predict_toxicity(request: TextRequest, threshold: float = 0.5):
    inputs = tokenizer(
        request.text,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        toxic_score = probs[0][1].item()

    return {
        "text": request.text,
        "toxicity_score": round(toxic_score, 4),
        "toxic": toxic_score >= threshold,
        "threshold_used": threshold
    }
