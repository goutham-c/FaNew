# 🕵️ Fake News Detector — DistilBERT

A fine-tuned transformer model that classifies news as **Real** or **Fake** with confidence scores.

Built with 🤗 Hugging Face Transformers, trained on the [`GonzaloA/fake_news`](https://huggingface.co/datasets/GonzaloA/fake_news) dataset (~44k articles).

---

## 📁 Project Structure

```
fake-news-detector/
├── train.py          # Fine-tune DistilBERT on the dataset
├── predict.py        # CLI inference tool
├── app.py            # Gradio web demo
├── requirements.txt  # Python dependencies
├── model/            # Saved model (created after training)
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train.py
```

> Training takes ~20–30 min on a GPU, or ~2–3 hrs on CPU.  
> The model is saved to `./model/` automatically.

### 3. Run Predictions (CLI)

```bash
# Interactive mode
python predict.py

# Single prediction
python predict.py "NASA confirms water found on Mars surface."
```

### 4. Launch the Web App

```bash
python app.py
# Opens at http://localhost:7860
```

---

## 🧠 Model Details

| Property         | Value                            |
|------------------|----------------------------------|
| Base model       | `distilbert-base-uncased`        |
| Task             | Binary text classification       |
| Labels           | `0 = Fake`, `1 = Real`           |
| Max input length | 256 tokens                       |
| Dataset          | GonzaloA/fake_news (~44k samples)|
| Epochs           | 3 (with early stopping)          |

**Expected performance** after training:

| Metric   | Value     |
|----------|-----------|
| Accuracy | ~97–98%   |
| F1 Score | ~0.97     |

---

## 📊 Dataset

Uses the [`GonzaloA/fake_news`](https://huggingface.co/datasets/GonzaloA/fake_news) dataset — a cleaned combination of several popular fake news corpora (ISOT, FakeNewsNet, etc.) hosted on Hugging Face Hub.

It is automatically downloaded during training via `datasets.load_dataset()`.

---

## 🚀 Publishing to GitHub

```bash
git init
git add .
git commit -m "Initial commit: DistilBERT fake news detector"
git remote add origin https://github.com/YOUR_USERNAME/fake-news-detector.git
git push -u origin main
```

Add your trained model weights too:
```bash
git lfs track "model/*"
git add model/
git commit -m "Add trained model weights"
git push
```

> 💡 Use [Git LFS](https://git-lfs.com/) for large model files.

---
"# FaNew" 
"# FaNew" 
