import streamlit as st
import joblib
import re

# -----------------------
# Load model and vectorizer
# -----------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    tfidf = joblib.load("tfidf.joblib")
    return model, tfidf

model, tfidf = load_model()
st.image("bg.jpg", use_column_width=True)
# -----------------------
# Text cleaning function
# -----------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------
# Streamlit app UI
# -----------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰")
st.title("📰 Fake News Detector")
st.write("Enter a news title and text to check if it's **Real or Fake**")

title = st.text_input("Enter News Title:")
text = st.text_area("Enter News Text:")

if st.button("Check News"):
    full_text = title + " " + text
    clean = clean_text(full_text)

    X_tfidf = tfidf.transform([clean])
    prediction = model.predict(X_tfidf)[0]

    if prediction == 0:
        st.success("✅ This looks like **REAL** news.")
    else:
        st.error("⚠️ This looks like **FAKE** news.")

