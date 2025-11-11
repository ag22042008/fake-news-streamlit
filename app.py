import streamlit as st
import joblib
import re
def set_background(image_file):
    page_bg_img = f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background-image: url("https://raw.githubusercontent.com/ag22042008/fake-news-streamlit/main/{image_file}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
    }}
    [data-testid="stSidebar"] {{
        background-color: rgba(255,255,255,0.6);
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)


set_background("bg.jpg") 
# -----------------------
# Load model and vectorizer
# -----------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    tfidf = joblib.load("tfidf.joblib")
    return model, tfidf

model, tfidf = load_model()

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


