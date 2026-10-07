"""
Fake News Detection - Prediction Script
Load the saved model and classify new articles from the command line.
"""

import sys
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_DIR = "./model"
MAX_LEN   = 256

def load_model():
    print(f"Loading model from {MODEL_DIR} ...")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model) -> dict:
    """Return label and confidence score for a piece of text."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0]
    label_id   = int(probs.argmax())
    label_name = model.config.id2label[label_id]
    confidence = float(probs[label_id]) * 100

    return {
        "label":      label_name.upper(),
        "confidence": round(confidence, 2),
        "fake_prob":  round(float(probs[0]) * 100, 2),
        "real_prob":  round(float(probs[1]) * 100, 2),
    }


def main():
    tokenizer, model = load_model()

    # If text passed as CLI arg, use it; otherwise enter interactive mode
    if len(sys.argv) > 1:
        text   = " ".join(sys.argv[1:])
        result = predict(text, tokenizer, model)
        print(f"\n📰 Input  : {text[:120]}...")
        print(f"🏷️  Label  : {result['label']}")
        print(f"🎯 Confidence: {result['confidence']}%")
        print(f"   Fake: {result['fake_prob']}%  |  Real: {result['real_prob']}%")
    else:
        print("Fake News Detector — Interactive Mode (type 'quit' to exit)\n")
        while True:
            text = input("Enter article text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            result = predict(text, tokenizer, model)
            print(f"\n  ➤ Label      : {result['label']}")
            print(f"  ➤ Confidence : {result['confidence']}%")
            print(f"  ➤ Fake: {result['fake_prob']}%  |  Real: {result['real_prob']}%\n")


if __name__ == "__main__":
    main()
