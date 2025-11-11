import streamlit as st
import joblib
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# ---------------------- #
#   Load model and TF-IDF
# ---------------------- #
try:
    model = joblib.load("model.joblib")
    tfidf = joblib.load("tfidf.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or TF-IDF: {e}")
    st.stop()

# ---------------------- #
#   NLTK setup
# ---------------------- #
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()

# ---------------------- #
#   Text cleaning function
# ---------------------- #
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation, numbers, symbols
    words = text.split()
    words = [ps.stem(word) for word in words if word not in STOPWORDS]
    return ' '.join(words)

# ---------------------- #
#   Streamlit page setup
# ---------------------- #
st.set_page_config(page_title="📰 Fake News Detection", page_icon="🧠", layout="centered")

# Add background styling
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
        background-color: rgba(255, 255, 255, 0.85);
        border-radius: 15px;
        padding: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- #
#   App title
# ---------------------- #
st.title("🧠 Fake News Detection App")
st.write("Enter a news headline or short article below to check if it's **Fake** or **Real**.")

# ---------------------- #
#   Input field
# ---------------------- #
user_input = st.text_area("📝 Enter News Text:", height=200, placeholder="Type your news article here...")

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to analyze.")
    else:
        cleaned = clean_text(user_input)
        vector_input = tfidf.transform([cleaned])
        prediction = model.predict(vector_input)[0]

        # Display result
        if prediction == 1:
            st.success("✅ The news appears to be **REAL**.")
        else:
            st.error("🚨 The news appears to be **FAKE**.")

# ---------------------- #
#   Footer
# ---------------------- #
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Developed by TEAM ⚡</p>",
    unsafe_allow_html=True
)
