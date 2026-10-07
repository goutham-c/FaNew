"""
Fake News Detection — Gradio Web App
Run:  python app.py
Then open  http://localhost:7860
"""

import gradio as gr
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

MODEL_DIR = "./model"
MAX_LEN   = 256

# ── Load model once at startup ────────────────────────────────────────────────
print("Loading model...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
model     = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()
print("Model ready ✅")


def classify(text: str):
    if not text.strip():
        return "⚠️ Please enter some text.", None

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
    )
    with torch.no_grad():
        logits = model(**inputs).logits

    probs      = torch.softmax(logits, dim=-1)[0]
    label_id   = int(probs.argmax())
    label_name = model.config.id2label[label_id]
    confidence = round(float(probs[label_id]) * 100, 2)

    fake_p = round(float(probs[0]) * 100, 2)
    real_p = round(float(probs[1]) * 100, 2)

    emoji  = "🔴 FAKE NEWS" if label_name == "fake" else "🟢 REAL NEWS"
    result = f"{emoji}  ({confidence}% confidence)"

    bar_data = {"Fake": fake_p, "Real": real_p}
    return result, bar_data


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Fake News Detector",
    theme=gr.themes.Soft(primary_hue="red", neutral_hue="slate"),
) as demo:

    gr.Markdown(
        """
        # 🕵️ Fake News Detector
        ### Powered by DistilBERT — fine-tuned for misinformation detection
        Paste a news headline or article excerpt below and click **Analyse**.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="News Text",
                placeholder="Paste a headline or article excerpt here...",
                lines=8,
            )
            analyse_btn = gr.Button("🔍 Analyse", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Label(label="Verdict")
            bar_output   = gr.BarPlot(
                x="Category",
                y="Probability (%)",
                title="Confidence Breakdown",
                height=220,
            )

    gr.Examples(
        examples=[
            ["Scientists discover that drinking coffee reverses all aging effects overnight."],
            ["The Federal Reserve raised interest rates by 0.25% following its latest policy meeting."],
            ["NASA confirms moon made entirely of cheese after new rover sample analysis."],
        ],
        inputs=text_input,
    )

    analyse_btn.click(
        fn=classify,
        inputs=text_input,
        outputs=[label_output, bar_output],
    )

if __name__ == "__main__":
    demo.launch(share=False)
