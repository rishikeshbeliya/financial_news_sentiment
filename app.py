import gradio as gr
import torch
import joblib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Loading the fine-tuned DistilBERT
bert_tokenizer = AutoTokenizer.from_pretrained("model/best_finance_sentiment_model")
bert_model = AutoModelForSequenceClassification.from_pretrained("model/best_finance_sentiment_model")
bert_model.eval()

# Load the TF-IDF + Logistic Regression baseline
tfidf_pipe = joblib.load("model/tfidf_lr_model.pkl")

labels = {0: "Neutral", 1: "Positive", 2: "Negative"}

def predict_bert(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    return {labels[i]: float(probs[i]) for i in range(3)}

def predict_tfidf(text):
    probs = tfidf_pipe.predict_proba([text])[0]
    return {labels[i]: float(probs[i]) for i in range(3)}

def compare_models(text):
    return predict_bert(text), predict_tfidf(text)

demo = gr.Interface(
    fn=compare_models,
    inputs=gr.Textbox(lines=3, placeholder="Paste a financial news here..."),
    outputs=[
        gr.Label(label="DistilBERT Prediction"),
        gr.Label(label="TF-IDF + Logistic Regression Prediction")
    ],
    title="Financial News Sentiment Analysis",
    description="Compare a fine-tuned DistilBERT model against a TF-IDF + LR baseline.",
    examples=[
        ["Central Bank Updates Online Archives Document Listing System\n\nThe Department of Banking Information Technologies completed a standard software update on its historical data archive servers this morning. The process involved updating file naming conventions for documents older than twenty years. No financial policies were reviewed, no economic metrics were released, and no changes were made to current regulatory guidelines. The update did not impact any active trading platforms or market operations."],
        ["Local Financial Tech Enterprise Secures Record Funding Round\n\nA local financial technology start-up successfully finalized an historic forty million dollar capital investment round this morning. The funding, led by prominent global venture firms, will allow the company to double its engineering workforce and expand operations immediately. Investors responded with immense enthusiasm, pushing regional tech sector valuations to new annual highs as optimism regarding the sector's localized economic growth potential accelerated."],
        ["Regional Logistics Firm Declares Sudden Bankruptcy Following Fraud\n\nA prominent domestic shipping and logistics corporation officially filed for emergency bankruptcy protection early this morning. The decision followed the sudden discovery of massive accounting irregularities and widespread institutional fraud across its regional distribution divisions. Share prices immediately crashed by ninety-four percent, completely erasing billions in shareholder value and triggering a broad wave of panic selling across the national transportation index."]
    ]
)

demo.launch()