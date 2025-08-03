import gradio as gr
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
import numpy as np

# Load TF-IDF Vectorizer
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# Load the classical ML models
lr_model = joblib.load("models/logistic_regression_tfidf.pkl")
svm_model = joblib.load("models/svm_tfidf_model.pkl")
nb_model = joblib.load("models/nb_tfidf_model.pkl")
rf_model = joblib.load("models/rf_tfidf_model.pkl")

# Load bert fine-tuned mmodel and tokenizer
model_name = "tarneemalaa/bert_imdb_model"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Prediction function
def predict_sentiment(model_picked, text, max_len=256):
    if not text or text.strip() == "":
        return "Please enter some text to analyze"
    
    # in case bert is chosen
    if model_picked == "BERT (Fine-tuned)":
        inputs = tokenizer(text, truncation=True, padding="max_length", max_length=max_len, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = output.logits
            probs = torch.softmax(logits, dim=1)
            pred_label = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_label].item()
            confidence_display = f"{confidence:.2%}"

    ## for the classical models   
    else: 
        vectorized = tfidf_vectorizer.transform([text])
            
        if model_picked == "Logistic Regression":
            probs = lr_model.predict_proba(vectorized)[0]
            pred_label = int(np.argmax(probs))
            confidence = probs[pred_label]
            confidence_display = f"{confidence:.2%}"
                
        elif model_picked == "SVM":
            pred_label = int(svm_model.predict(vectorized)[0])
            confidence_display = "<i>Not available for SVM</i>"
                
        elif model_picked == "Naive Bayes":
            probs = nb_model.predict_proba(vectorized)[0]
            pred_label = int(np.argmax(probs))
            confidence = probs[pred_label]
            confidence_display = f"{confidence:.2%}"
                
        elif model_picked == "Random Forest": 
            probs = rf_model.predict_proba(vectorized)[0]
            pred_label = int(np.argmax(probs))
            confidence = probs[pred_label]
            confidence_display = f"{confidence:.2%}"
            

    sentiment = "Positive" if pred_label == 1 else "Negative"
    emoji = "✅" if sentiment == "Positive" else "❌"
    color = "green" if sentiment == "Positive" else "red"

    return f"""
        <div style="font-size: 24px; font-weight: bold; color: {color}; margin-bottom: 10px;">
            {emoji} Sentiment: {sentiment}
        </div>
        <div style="font-size: 18px; color: #666;">
            Confidence: {confidence_display}
        </div>
        """

demo = gr.Interface(
    fn=predict_sentiment,
    inputs=[
        gr.Dropdown(
            choices=[
                "BERT (Fine-tuned)",
                "Logistic Regression", 
                "SVM",
                "Naive Bayes",
                "Random Forest"
            ],
            label="Choose Model",
            value="BERT (Fine-tuned)"
        ),
        gr.Textbox(lines=6, placeholder="Paste a movie review here...", label="🎬 Movie Review")
    ],
    outputs=gr.HTML(label="Prediction Result"),
    title="🎬 IMDb Sentiment Classifier",
    description="This app allows you to **compare** a **fine-tuned BERT** model with **classical ML models** (Logistic Regression, SVM, Naive Bayes, Random Forest) on IMDb movie reviews.\n\nMade by [Tarneem Alaa](https://github.com/tarneemalaa1)",
    theme=gr.themes.Soft(),
    examples=[
        ["BERT (Fine-tuned)", "This movie was absolutely amazing, I enjoyed every moment of it!"],
        ["Logistic Regression", "It was a total waste of time. The plot made no sense."],
        ["SVM", "Great acting and wonderful storyline. Highly recommend!"],
        ["Naive Bayes", "Boring and predictable. Not worth watching."]
    ],
    flagging_mode="never"
)

if __name__ == "__main__":
    demo.launch()