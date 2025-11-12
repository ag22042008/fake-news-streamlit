import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
import os

# ---------------------- #
#   Load model and TF-IDF
# ---------------------- #
try:
    model = joblib.load("model.joblib")
    tfidf = joblib.load("tfidf.joblib")
except Exception as e:
    st.error(f"❌ Error loading model or TF-IDF: {e}")
    st.stop()

# ---------------------- #
#   NLTK setup
# ---------------------- #
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))

# ---------------------- #
#   Text cleaning function (no stemming)
# ---------------------- #
def clean_text(text):
    # convert to string and lowercase
    text = str(text).lower()
    # remove punctuation and numbers (keep spaces and letters)
    text = re.sub(r'[^a-z\s]', ' ', text)
    # collapse multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # remove stopwords
    words = [w for w in text.split() if w not in STOPWORDS]
    return ' '.join(words)

# ---------------------- #
#   Streamlit page setup
# ---------------------- #
st.set_page_config(page_title="📰 Fake News Detection", page_icon="🧠", layout="centered")

# Add background styling (optional; change URL or remove)
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-image: url("https://images.unsplash.com/photo-1522199710521-72d69614c702?fit=crop&w=1200&q=80");
        background-size: cover;
        background-position: center;
    }
    [data-testid="stHeader"] {
        background: rgba(0,0,0,0);
    }
    .stApp {
        background-color: rgba(255, 255, 255, 0.88);
        border-radius: 12px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- #
#   App title & input
# ---------------------- #
st.title("🧠 Fake News Detection (No Stemming)")
st.write("Paste a news headline or short article below to check if it's likely **Fake** or **Real**.")

user_input = st.text_area("📝 Enter News Text:", height=200, placeholder="Type your news article or headline here...")

# ---------------------- #
#   Predict button
# ---------------------- #
if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        try:
            vector_input = tfidf.transform([cleaned])
            prediction = model.predict(vector_input)[0]
            # Optional: probability/confidence if model supports predict_proba
            confidence = None
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vector_input)[0]
                # if binary (2 classes), take max prob
                confidence = max(probs)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            st.stop()

        # Display result
        if prediction == 1:
            st.success("✅ The news appears to be **REAL**.")
        else:
            st.error("🚨 The news appears to be **FAKE**.")

        if confidence is not None:
            st.write(f"Confidence: **{confidence:.2%}**")

# ---------------------- #
#   Footer
# ---------------------- #
st.markdown("---")
st.markdown("<p style='text-align:center; color:gray;'>Developed by Aditya Gupta ⚡</p>", unsafe_allow_html=True)


