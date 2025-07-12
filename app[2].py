import streamlit as st
import pickle
import re
import nltk
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
nltk.download('stopwords')

# ---------------- Page Setup ---------------- #
st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="centered")

# Background image URL (update if needed)
background_url = "https://images.unsplash.com/photo-1581093588401-12cbf1b01956?auto=format&fit=crop&w=1050&q=80"

# ---------------- Custom CSS ---------------- #
st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("{background_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .main {{
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }}
    .stButton > button {{
        background-color: #0073e6;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.6em 1.2em;
    }}
    .stTextArea > div > textarea {{
        font-size: 16px;
    }}
    h1, h2, h3 {{
        color: #003366;
    }}
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Model ---------------- #
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- Text Preprocessor ---------------- #
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    return text

# ---------------- Session State Init ---------------- #
if 'admin_logged_in' not in st.session_state:
    st.session_state['admin_logged_in'] = False
if 'history' not in st.session_state:
    st.session_state['history'] = pd.DataFrame(columns=["Review", "Prediction", "Confidence"])

# ---------------- Sidebar Login ---------------- #
with st.sidebar:
    st.title("🔐 Admin Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.success("Login Successful")
            st.session_state['admin_logged_in'] = True
        else:
            st.error("Invalid credentials")

# ---------------- Main Title ---------------- #
st.title("🕵️ Fake Review Detection System")

# ---------------- Admin Dashboard ---------------- #
if st.session_state['admin_logged_in']:
    st.subheader("📊 Admin Dashboard")
    if len(st.session_state['history']) > 0:
        st.write("### 🔁 Recent Predictions")
        st.dataframe(st.session_state['history'].iloc[::-1].reset_index(drop=True))
        st.download_button("📥 Download History as CSV",
                           data=st.session_state['history'].to_csv(index=False),
                           file_name="prediction_history.csv")
    else:
        st.info("No predictions yet.")
    if st.button("Logout"):
        st.session_state['admin_logged_in'] = False

# ---------------- User Input ---------------- #
else:
    st.write("### ✍️ Enter a review to detect if it's fake or genuine")
    input_text = st.text_area("Your review here...", height=150)
    if st.button("Analyze Review"):
        if input_text.strip() == "":
            st.warning("Please enter a review to analyze.")
        else:
            cleaned = clean_text(input_text)
            vec = vectorizer.transform([cleaned]).toarray()
            result = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]

            # Fix label mapping (0 = genuine, 1 = fake)
            prediction = "✅ Genuine Review" if result == 0 else "⚠️ Fake Review"
            confidence = f"{np.max(proba)*100:.2f}%"

            st.markdown(f"### 🎯 Prediction: {prediction}")
            st.markdown(f"**🧠 Confidence Score:** `{confidence}`")
            st.bar_chart({"Genuine": proba[0], "Fake": proba[1]})

            new_row = {"Review": input_text, "Prediction": prediction, "Confidence": confidence}
            st.session_state['history'] = pd.concat(
                [st.session_state['history'], pd.DataFrame([new_row])],
                ignore_index=True
            )

    # ---------------- Sample Inputs ---------------- #
    with st.expander("🧪 Try Sample Reviews"):
        test_reviews = [
            "Absolutely terrible product. Worst I’ve used.",
            "Great product, very useful. Loved it!",
            "Fake review. Totally not real at all.",
            "This is genuine. I actually bought and used it.",
            "Highly recommended. Excellent quality!"
        ]
        for rev in test_reviews:
            cleaned = clean_text(rev)
            vec = vectorizer.transform([cleaned])
            result = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            pred = "✅ Genuine" if result == 0 else "⚠️ Fake"
            st.markdown(f"**Review:** {rev}")
            st.markdown(f"Prediction: `{pred}` | Confidence: `{np.max(proba)*100:.2f}%`")
            st.progress(float(np.max(proba)))
