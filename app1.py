import streamlit as st
import numpy as np
import re
import pickle
from gensim.models import KeyedVectors
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

w2v_model = KeyedVectors.load(
    os.path.join(BASE_DIR, "w2v_vectors.kv"),
    mmap='r'
)

# =========================
# Load trained models
# =========================
lr_model = pickle.load(open("logistic_w2v_model.pkl", "rb"))
w2v_model = Word2Vec.load("w2v_model.model")

# =========================
# Text preprocessing
# =========================
def preprocess_text(review):
    review = re.sub(r'<.*?>', '', review)   # remove HTML
    review = review.lower()
    review = re.sub(r'[^a-z\s]', '', review)
    review = re.sub(r'\s+', ' ', review).strip()
    return review

# =========================
# Negation handling
# =========================
def mark_negation(review):
    words = review.split()
    negation_words = ["not", "no", "never", "none", "n't"]
    negated = False
    new_words = []

    for word in words:
        if word in negation_words:
            negated = True
            new_words.append(word)
        elif negated:
            new_words.append(word + "_NEG")
        else:
            new_words.append(word)

    return " ".join(new_words)

# =========================
# Sentence to vector
# =========================
def sentence_vector(sentence, model, size=100):
    words = sentence.split()
    vec = np.zeros(size)
    count = 0

    for word in words:
        if word in model.wv:
            vec += model.wv[word]
            count += 1

    if count != 0:
        vec /= count

    return vec.reshape(1, -1)

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("🎬 IMDB Sentiment Analysis")
st.write("**Word2Vec + Logistic Regression**")

user_input = st.text_area("Enter a movie review:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean = preprocess_text(user_input)
        neg = mark_negation(clean)
        vec = sentence_vector(neg, w2v_model)

        pred = lr_model.predict(vec)[0]
        prob = lr_model.predict_proba(vec)[0]

        sentiment = "Positive 😊" if pred == 1 else "Negative 😞"

        st.subheader("Prediction")
        st.write(sentiment)

        st.subheader("Confidence Scores")
        st.write(f"Negative: {prob[0]:.4f}")
        st.write(f"Positive: {prob[1]:.4f}")
