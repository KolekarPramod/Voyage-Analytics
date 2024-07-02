import streamlit as st
import joblib
import pandas as pd

# Load the model
knn_model = joblib.load('knn_model.pkl')

# Define function to make predictions
def predict(company, flightType, age):
    data = {
        'company': [company],
        'flightType': [flightType],
        'age': [age]
    }
    df = pd.DataFrame(data)
    prediction = knn_model.predict(df)
    return prediction[0]

# Streamlit app
st.title('Gender Prediction App')

# Get user input
company = st.selectbox('Company', options=[0, 1, 2])  # Adjust options based on your data
flightType = st.selectbox('Flight Type', options=[0, 1])  # Adjust options based on your data
age = st.number_input('Age', min_value=0, max_value=100)

# Make prediction
if st.button('Predict'):
    gender = predict(company, flightType, age)
    gender_label = 'Male' if gender == 1 else 'Female'
    st.write(f'Predicted Gender: {gender_label}')
