import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from transformers import pipeline
import hashlib

# Set up page
st.set_page_config(page_title="Fake Review Detector (BERT)", page_icon="🔍")

# Load BERT pipeline
@st.cache_resource
def load_model():
    return pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

classifier = load_model()

# Hashed password check
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_HASH = "e99a18c428cb38d5f260853678922e03"  # hash for 'admin123'

# Sidebar login
st.sidebar.title("Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
admin_logged_in = username == ADMIN_USERNAME and hash_password(password) == ADMIN_PASSWORD_HASH

st.title("🤖 Fake Review Detector (BERT Model)")

if admin_logged_in:
    st.success("Logged in as Admin")
    st.subheader("Admin Dashboard")
    if 'history' in st.session_state:
        st.write("### Recent Predictions")
        st.dataframe(st.session_state['history'])
        st.download_button(
            "Download History as CSV",
            st.session_state['history'].to_csv(index=False),
            "prediction_history.csv",
            "text/csv"
        )
    else:
        st.info("No prediction history yet.")
else:
    input_text = st.text_area("Enter a product or hotel review")
    if st.button("Analyze"):
        with st.spinner("Analyzing using BERT..."):
            result = classifier(input_text)[0]
            label = result['label']
            score = result['score']
            prediction = "✅ Genuine Review" if label == "REAL" else "⚠️ Fake Review"
            confidence = f"{score*100:.2f}%"

            st.write("## 🎯 Prediction:")
            st.success(prediction)
            st.write(f"**Confidence Score:** {confidence}")
            st.bar_chart({"Confidence": [score], "Uncertainty": [1 - score]})

            # Save to history
            if 'history' not in st.session_state:
                st.session_state['history'] = pd.DataFrame(columns=["Review", "Prediction", "Confidence", "Timestamp"])
            new_row = {
                "Review": input_text,
                "Prediction": prediction,
                "Confidence": confidence,
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            st.session_state['history'] = pd.concat(
                [st.session_state['history'], pd.DataFrame([new_row])], ignore_index=True)
