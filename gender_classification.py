import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import joblib
import os

def gender_classification_page():
    # st.set_page_config(page_title="Name Gender Classifier", layout="wide")
    st.title("Name-Based Gender Prediction")

    model_file = './models/name_gender_classifier.pkl'

    # Check file existence
    if not os.path.exists(model_file):
        st.error("Model file not found! Please train the model first using `train_gender_classifier.py`.")
        return

    try:
        # Load full dict with classifier + label encoder
        model_data = joblib.load(model_file)
        classifier = model_data['classifier']
        label_encoder = model_data['label_encoder']
    except Exception as e:
        st.error(f"Failed to load model data: {e}")
        return

    # Sentence Transformer
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # User input
    name = st.text_input("Enter a name:", "")

    if name:
        with st.spinner("Predicting gender..."):
            embedding = model.encode([name])

            prediction = classifier.predict(embedding)
            probability = classifier.predict_proba(embedding)
            gender = label_encoder.inverse_transform(prediction)[0]

            confidence = max(probability[0]) * 100
            st.metric("Predicted Gender", gender, f"{confidence:.1f}% confidence")

if __name__ == "__main__":
    gender_classification_page()
