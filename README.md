# Financial News Sentiment Analysis

Classifies financial news as **Positive / Negative / Neutral**, comparing a fine-tuned **DistilBERT** model against a **TF-IDF + Logistic Regression** baseline. Deployed with Gradio on HuggingFace Spaces.

🔗 **Live demo:** [huggingface.co/spaces/rishikeshbeliya/financial-news-sentiment](https://huggingface.co/spaces/rishikeshbeliya/financial-news-sentiment)

## Problem Statement

Financial news sentiment drives short-term market reactions. This project explores whether a fine-tuned transformer model meaningfully outperforms a classical NLP baseline on this task — useful context for anomaly/event detection pipelines.

## Dataset

[Add Kaggle dataset name + link], split 80/10/10 into train/cv/test with stratified sampling on sentiment label.

## Models Compared

| Model | Test Accuracy | Test Macro F1 |
|---|---|---|
| DistilBERT (fine-tuned) | 81% | 78% |
| TF-IDF + Logistic Regression | 80% | 76% |

[1-2 sentences on which performed better and why, e.g. transformer captures context vs. baseline relies on keyword frequency.]

## Tech Stack

Python, PyTorch, HuggingFace Transformers, scikit-learn, Gradio, HuggingFace Spaces

## Project Structure

```
financial_news_sentiment/
├── app.py
├── requirements.txt
├── notebook/
│   └── training_notebook.ipynb
├── data/
└── model/
    ├── best_finance_sentiment_model/
    └── tfidf_lr_model.pkl
```

## Run Locally

```bash
git clone https://github.com/rishikeshbeliya/financial_news_sentiment.git
cd financial_news_sentiment
pip install -r requirements.txt
python app.py
```

## Future Improvements

- Auto-label live news via NewsAPI for real-time demo
- FastAPI deployment as a REST endpoint
- SHAP explainability for the TF-IDF baseline
