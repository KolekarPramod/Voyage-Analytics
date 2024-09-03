import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder

def gender_classification_page():
    st.title("Gender Classification Model")

    # Initialize the SentenceTransformer model
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

    # Load the trained classification model and scaler model
    scaler_model = pickle.load(open("scaler.pkl", 'rb'))
    pca_model = pickle.load(open("pca.pkl", 'rb'))
    logistic_model = pickle.load(open("tuned_logistic_regression_model.pkl", 'rb'))

    # Set default values for Usercode, Traveller Age, and Company Name (Because it doesn't contribute into prediction of gender.)
    usercode = 0
    traveller_age = 21
    company_name = "Acme Factory"

    # Only show the input for the username
    username = st.text_input("Username", "Charlotte Johnson")

    if st.button("Predict"):
        # Create a dictionary to store the input data
        data = {
            'code': usercode,
            'company': company_name,
            'name': username,
            'age': traveller_age,
        }

        # Perform prediction using the custom_input dictionary
        prediction = predict_price(data, logistic_model, pca_model, scaler_model)

        if prediction == 0:
            gender = 'female'
        else:
            gender = 'male'

        st.success(f"Predicted gender: {gender}")

def predict_price(input_data, lr_model, pca, scaler):
    # Prepare the input data
    text_columns = ['name']
    model = SentenceTransformer('flax-sentence-embeddings/all_datasets_v4_MiniLM-L6')

    # Initialize an empty DataFrame
    df = pd.DataFrame([input_data])

    # Encode userCode and company to numeric values
    label_encoder = LabelEncoder()

    df['company_encoded'] = label_encoder.fit_transform(df['company'])

    # Encode text-based columns and create embeddings
    for column in text_columns:
        df[column + '_embedding'] = df[column].apply(lambda text: model.encode(text))

    # Apply PCA separately to each text embedding column
    n_components = 23  # Adjust the number of components as needed
    text_embeddings_pca = np.empty((len(df), n_components * len(text_columns)))

    for i, column in enumerate(text_columns):
        embeddings = df[column + '_embedding'].values.tolist()
        embeddings_pca = pca.transform(embeddings)
        text_embeddings_pca[:, i * n_components:(i + 1) * n_components] = embeddings_pca

    # Combine text embeddings with other numerical features if available
    numerical_features = ['code', 'company_encoded', 'age']

    X_numerical = df[numerical_features].values

    # Combine PCA-transformed text embeddings and numerical features
    X = np.hstack((text_embeddings_pca, X_numerical))

    # Scale the data using the same scaler used during training
    X = scaler.transform(X)

    # Make predictions using the trained Linear Regression model
    y_pred = lr_model.predict(X)

    return y_pred[0]

# Run the app
if __name__ == '__main__':
    gender_classification_page()
