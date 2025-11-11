import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load model and TF-IDF
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# Page configuration
st.set_page_config(page_title="Fake News Detector 📰", page_icon="🗞️", layout="centered")

# Download NLTK stopwords (if not already downloaded)
nltk.download('stopwords')

ps = PorterStemmer()

# Cleaning function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.split()
    text = [ps.stem(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# 🎨 Custom CSS styling
st.markdown("""
    <style>
    /* Background */
    .stApp {
        background-image: url('https://images.unsplash.com/photo-1525182008055-f88b95ff7980');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }

    /* Card container */
    .main {
        background-color: rgba(255, 255, 255, 0.88);
        border-radius: 20px;
        padding: 2rem 2.5rem;
        box-shadow: 0px 4px 20px rgba(0, 0, 0, 0.25);
        margin-top: 3rem;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Headings */
    h1, h2, h3 {
        color: #222;
        text-align: center;
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }

    /* Buttons */
    div.stButton > button {
        background-color: #0078FF;
        color: white;
        border-radius: 12px;
        height: 3rem;
        width: 100%;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        transition: 0.3s ease;
    }
    div.stButton > button:hover {
        background-color: #005FCC;
        transform: scale(1.03);
    }

    /* Text area */
    textarea {
        border-radius: 10px !important;
        border: 2px solid #0078FF !important;
        background-color: #f9f9f9 !important;
    }

    /* Hide footer */
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# 🌐 Main content
st.markdown("<div class='main'>", unsafe_allow_html=True)

st.title("📰 Fake News Detection App")
st.subheader("Check if a news headline or article is Real or Fake")

# User input
user_input = st.text_area("Enter News Text Below:", height=150)

if st.button("🔍 Analyze"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vector_input = tfidf.transform([cleaned])
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.success("✅ This looks like **Real News**.")
        else:
            st.error("❌ This appears to be **Fake News**.")

st.markdown("</div>", unsafe_allow_html=True)
