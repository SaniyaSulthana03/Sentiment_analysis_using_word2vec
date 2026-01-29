import streamlit as st
import numpy as np
import re
import pickle
import os

from gensim.models import KeyedVectors

# =========================
# App Config
# =========================
st.set_page_config(
    page_title="Sentiment Analysis (Word2Vec + Logistic Regression)",
    layout="centered"
)

st.title("🎬 Sentiment Analysis using Word2Vec")
st.write("Logistic Regression | IMDB Reviews")

# =========================
# Load Models
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models():
    lr_model = pickle.load(
        open(os.path.join(BASE_DIR, "logistic_w2v_model.pkl"), "rb")
    )

    w2v_model = KeyedVectors.load(
        os.path.join(BASE_DIR, "w2v_vectors.kv"),
        mmap="r"
    )

    return lr_model, w2v_model


lr_model, w2v_model = load_models()

VECTOR_SIZE = w2v_model.vector_size

# =========================
# Text Preprocessing
# =========================
def preprocess_text(text):
    text = re.sub(r"<.*?>", "", text)       # remove HTML tags
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)    # remove punctuation & numbers
    text = re.sub(r"\s+", " ", text).strip()
    return text


def mark_negation(text):
    words = text.split()
    negation_words = {"not", "no", "never", "none", "n't"}
    negated = False
    new_words = []

    for word in words:
        if word in negation_words:
            negated = True
            new_words.append(word)
        elif re.search(r"[.!?]", word):
            negated = False
            new_words.append(word)
        elif negated:
            new_words.append(word + "_NEG")
        else:
            new_words.append(word)

    return " ".join(new_words)


def sentence_vector(sentence, model):
    words = sentence.split()
    vec = np.zeros(VECTOR_SIZE)
    count = 0

    for w in words:
        if w in model:
            vec += model[w]
            count += 1

    if count > 0:
        vec /= count

    return vec.reshape(1, -1)


def predict_sentiment(text):
    clean = preprocess_text(text)
    neg_text = mark_negation(clean)
    vector = sentence_vector(neg_text, w2v_model)

    prediction = lr_model.predict(vector)[0]
    probabilities = lr_model.predict_proba(vector)[0]

    sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"

    return sentiment, probabilities


# =========================
# Streamlit UI
# =========================
user_input = st.text_area(
    "✍️ Enter a movie review:",
    height=150,
    placeholder="Example: I do not like this movie at all..."
)

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        sentiment, probs = predict_sentiment(user_input)

        st.subheader("Prediction")
        st.success(f"Sentiment: **{sentiment}**")

        st.subheader("Confidence Scores")
        st.write(f"Negative: **{probs[0]:.4f}**")
        st.write(f"Positive: **{probs[1]:.4f}**")

        st.progress(float(probs[1]))

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "Built with Word2Vec embeddings & Logistic Regression | Streamlit Deployment"
)
