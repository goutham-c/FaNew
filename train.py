"""
Fake News Detection - Training Script
Uses DistilBERT fine-tuned on the LIAR or WELFake dataset from Hugging Face
"""

import os
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_NAME   = "distilbert-base-uncased"
DATASET_NAME = "GonzaloA/fake_news"   # HuggingFace dataset (real=1, fake=0)
OUTPUT_DIR   = "./model"
MAX_LEN      = 256
BATCH_SIZE   = 16
EPOCHS       = 3
LR           = 2e-5
SEED         = 42

LABEL2ID = {"fake": 0, "real": 1}
ID2LABEL = {0: "fake", 1: "real"}

# ── Load tokenizer ───────────────────────────────────────────────────────────
print("Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


def load_and_prepare():
    """Download dataset and return train/val/test splits."""
    print(f"Loading dataset: {DATASET_NAME}")
    ds = load_dataset(DATASET_NAME)

    # Some HF fake-news datasets have a 'label' column (0=fake, 1=real)
    # and a 'text' or combined title+text column.
    def combine_text(example):
        title = example.get("title", "") or ""
        text  = example.get("text",  "") or ""
        example["combined"] = (title + " " + text).strip()
        return example

    ds = ds.map(combine_text)

    # Split train → train + validation (90/10)
    split = ds["train"].train_test_split(test_size=0.1, seed=SEED)
    return split["train"], split["test"], ds.get("test")


def tokenize(batch):
    return tokenizer(
        batch["combined"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }


def main():
    train_raw, val_raw, test_raw = load_and_prepare()

    print("Tokenising splits...")
    cols = ["input_ids", "attention_mask", "label"]

    train_ds = train_raw.map(tokenize, batched=True).rename_column("label", "labels")
    val_ds   = val_raw.map(tokenize,   batched=True).rename_column("label", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch",   columns=["input_ids", "attention_mask", "labels"])

    # ── Model ────────────────────────────────────────────────────────────────
    print("Loading DistilBERT model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # ── Training args ─────────────────────────────────────────────────────────
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LR,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="./logs",
        logging_steps=50,
        seed=SEED,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    print("\n🚀 Starting training...")
    trainer.train()

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    if test_raw is not None:
        print("\nEvaluating on test set...")
        test_ds = test_raw.map(tokenize, batched=True).rename_column("label", "labels")
        test_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        results = trainer.evaluate(test_ds)
        print("Test results:", results)

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\n✅ Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done! Model saved.")


if __name__ == "__main__":
    main()
