import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd

# ---------------- Page Setup ---------------- #
st.set_page_config(page_title="Fake Review Detector", page_icon="🔍", layout="centered")

st.markdown("""
    <style>
        .main {background-color: #f7f9fc;}
        .stButton > button {
            background-color: #0073e6;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 0.6em 1.2em;
        }
        .stTextArea > div > textarea {
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Model ---------------- #
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ---------------- Text Cleaner ---------------- #
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    tokens = text.split()
    ps = PorterStemmer()
    tokens = [ps.stem(word) for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

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
        st.download_button("📥 Download History as CSV", data=st.session_state['history'].to_csv(index=False),
                           file_name="prediction_history.csv")
    else:
        st.info("No predictions yet.")
    st.button("Logout", on_click=lambda: st.session_state.update({'admin_logged_in': False}))

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
            prediction = "✅ Genuine Review" if result == 1 else "⚠️ Fake Review"
            confidence = f"{np.max(proba)*100:.2f}%"

            # Result display
            st.markdown(f"### 🎯 Prediction: {prediction}")
            st.markdown(f"**🧠 Confidence Score:** `{confidence}`")
            st.bar_chart({"Genuine": proba[1], "Fake": proba[0]})

            # Store result
            new_row = {"Review": input_text, "Prediction": prediction, "Confidence": confidence}
            st.session_state['history'] = pd.concat(
                [st.session_state['history'], pd.DataFrame([new_row])],
                ignore_index=True
            )
