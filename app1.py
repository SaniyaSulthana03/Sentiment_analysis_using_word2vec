import streamlit as st
import pandas as pd
import numpy as np
import re
import pickle
from gensim.models import Word2Vec

# =========================
# Page config
# =========================
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("🎬 Sentiment Analysis using Word2Vec + Logistic Regression")
st.write("Predict sentiment with confidence score (handles negations correctly).")

# =========================
# Text preprocessing
# =========================
def preprocess_text(text):
    text = re.sub(r"<.*?>", "", text)          # remove HTML
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)       # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text

def mark_negation(text):
    words = text.split()
    neg_words = {"not", "no", "never", "none", "n't"}
    negated = False
    result = []

    for w in words:
        if w in neg_words:
            negated = True
            result.append(w)
        elif negated:
            result.append(w + "_NEG")
        else:
            result.append(w)

    return " ".join(result)

def sentence_vector(sentence, wv, size=100):
    vec = np.zeros(size)
    count = 0
    for w in sentence.split():
        if w in wv:
            vec += wv[w]
            count += 1
    if count > 0:
        vec /= count
    return vec.reshape(1, -1)

# =========================
# Load model & train Word2Vec
# =========================
@st.cache_resource
def load_models():
    # Load trained Logistic Regression model
    with open("logistic_w2v_model.pkl", "rb") as f:
        lr_model = pickle.load(f)

    # Load dataset to train Word2Vec
    df = pd.read_csv("IMDB Dataset.csv")

    # Preprocess + negation
    df["processed"] = df["review"].apply(
        lambda x: mark_negation(preprocess_text(x))
    )

    sentences = [text.split() for text in df["processed"]]

    # Train Word2Vec inside app
    w2v_model = Word2Vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=2,
        workers=4
    )

    return lr_model, w2v_model.wv

lr_model, wv = load_models()

# =========================
# User input
# =========================
user_text = st.text_area("Enter your review:", height=150)

if st.button("Predict Sentiment"):
    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        clean = preprocess_text(user_text)
        neg = mark_negation(clean)
        vec = sentence_vector(neg, wv)

        pred = lr_model.predict(vec)[0]
        prob = lr_model.predict_proba(vec)[0]

        sentiment = "Positive 😊" if pred == 1 else "Negative 😞"

        st.subheader("Prediction")
        st.success(sentiment)

        st.subheader("Confidence Scores")
        st.write(f"Negative: {prob[0]:.4f}")
        st.write(f"Positive: {prob[1]:.4f}")
