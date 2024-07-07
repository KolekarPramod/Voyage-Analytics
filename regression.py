# regression.py
import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

def main():
    st.title('Flight Price Predictor')

    # Load the trained model
    model = joblib.load('flight_price_model (1).joblib')

    from_location = st.selectbox('From Location', ['Recife (PE)', 'Florianopolis (SC)', 'Brasilia (DF)',
           'Aracaju (SE)', 'Salvador (BH)', 'Campo Grande (MS)',
           'Sao Paulo (SP)', 'Natal (RN)', 'Rio de Janeiro (RJ)'])
    to_location = st.selectbox('To Location', ['Florianopolis (SC)', 'Recife (PE)', 'Brasilia (DF)',
           'Salvador (BH)', 'Aracaju (SE)', 'Campo Grande (MS)',
           'Sao Paulo (SP)', 'Natal (RN)', 'Rio de Janeiro (RJ)'])
    flightType = st.selectbox('Flight Type', ['firstClass', 'economy','premium'])
    agency = st.selectbox('Agency', ['FlyingDrops', 'CloudFy','Rainbow'])
    time = st.number_input('Time (in hours)', min_value=0.0, value=1.76)
    distance = st.number_input('Distance (in km)', min_value=0.0, value=676.53)
    date = st.date_input('Date', datetime(2019, 9, 26))

    # Preprocess input
    input_data = pd.DataFrame({
        # 'travelCode': [travelCode],
        # 'userCode': [userCode],
        'from': [from_location],
        'to': [to_location],
        'flightType': [flightType],
        'agency': [agency],
        'time': [time],
        'distance': [distance],
        'day': [date.day],
        'month': [date.month],
        'year': [date.year]
    })

    # Predict flight price
    if st.button('Predict Price'):
        price = model.predict(input_data)
        st.write(f'Predicted Flight Price: ${price[0]:.2f}')

if __name__ == '__main__':
    main()
