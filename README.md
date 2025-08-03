# 🎬 IMDb Sentiment Classifier – Classical ML vs. BERT

This project is an **end-to-end sentiment analysis pipeline** built using both **classical machine learning models** and a **fine-tuned BERT transformer** on the IMDb movie reviews dataset.

It includes: 
 - Text Preprocessing
 - CountVectorizer & TF-IDF Vectorization
 - Classical Machine Learning Models: Logistic Regression, SVM, Naive Bayes and Random Forest
 - Fine-tuned BERT
 - Gradio App for real time sentiment analysis for movie reviews
 - Deployed on Hugging Face Spaces

--- 

## Try the App from Here
Test the models on your own reviews and compare results in real time

- [**Light Mode App**](https://tarneemalaa-imdb-sentiment-classifier.hf.space/?__theme=light)
- [**Dark Mode App**](https://tarneemalaa-imdb-sentiment-classifier.hf.space/?__theme=dark)

You can enter the movie review here along with choosing the model and enjoy! You can also compare performances of different models on the same review.

---

## Blog Post 
I wrote a very detailed blog post on hashnode for this project that you can check out.

- [**Building an End-to-End Sentiment Classifier – Classical ML vs. BERT on IMDb Reviews**](https://tarneem.hashnode.dev/building-an-end-to-end-sentiment-classifier-classical-ml-vs-bert-on-imdb-review)


---

## Models Results

| Model                   | Train Accuracy | Val Accuracy | Test Accuracy | Training Time |
|------------------------|----------------|---------------|----------------|----------------|
| **BERT**               | 98.33%         | 92.61%        | **93.08%**        | 4818.46s       |
| Logistic Regression    | 92.40%         | 89.04%        | 89.83%         | 0.58s          |
| SVM                    | 96.37%         | 88.32%        | 89.11%         | 0.85s          |
| Naive Bayes            | 88.09%         | 86.23%        | 87.33%         | 0.02s          |
| Random Forest          | 100.00%        | 85.00%        | 85.87%         | 150.90s        |

---

## Fine-tuned BERT Model
You can find the fine-tuned BERT model uploaded on HuggingFace here: 
[Fine-tuned BERT IMDb Model](https://huggingface.co/tarneemalaa/bert_imdb_model)

You can use it like this:

```python
from transformers import BertTokenizer, BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained("tarneemalaa/bert_imdb_model")
tokenizer = BertTokenizer.from_pretrained("tarneemalaa/bert_imdb_model")
```

---

## Installation and Running

git clone https://github.com/TarneemAlaa1/imdb-sentiment-classifier

cd imdb-sentiment-classifier

pip install -r requirements.txt

cd app

python app.py

---

## Contact Me

- [LinkedIn](https://www.linkedin.com/in/tarneem-alaa-abdelreheem/)
- [Github](https://github.com/TarneemAlaa1)
- [HuggingFace](https://huggingface.co/tarneemalaa)
- [Hashnode](https://tarneem.hashnode.dev/)