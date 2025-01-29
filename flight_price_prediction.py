import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the saved model and encoders
model = joblib.load('flight_price_model.joblib')
le_from = joblib.load('le_from.joblib')
le_to = joblib.load('le_to.joblib')
le_type = joblib.load('le_type.joblib')
le_agency = joblib.load('le_agency.joblib')

# Load the CSV file containing time and distance information
new_df = pd.read_csv('./new_df.csv')

def flight_price_prediction_page():
    """Function for the Flight Price Prediction page."""
    # Set page title
    st.title('Flight Price Predictor')

    # Create input form
    st.header('Enter Flight Details')

    # Get unique values for dropdowns
    from_cities = le_from.classes_
    to_cities = le_to.classes_
    
    # Ensure 'From' and 'To' are not the same
    from_location = st.selectbox("Select Departure Location", new_df['from'].unique())
    to_location = st.selectbox("Select Arrival Location", new_df['to'].unique())

    if from_location == to_location:
        st.write("Departure and Arrival locations cannot be the same. Please select different locations.")
    else:
        # Automatically fill 'time' and 'distance' based on selected locations
        filtered_row = new_df[(new_df['from'] == from_location) & (new_df['to'] == to_location)]
        if not filtered_row.empty:
            time = filtered_row.iloc[0]['time']
            distance = filtered_row.iloc[0]['distance']
        else:
            st.warning("No data available for the selected route.")
            time, distance = 0, 0

        # Display time and distance to the user
        st.write(f"Time: {time} hours")
        st.write(f"Distance: {distance} km")

        # Dropdown for flight type
        flight_types = ['economic', 'premium', 'firstClass']
        flight_type = st.selectbox('Flight Type:', flight_types)

        # Dropdown for agency
        agencies = le_agency.classes_
        agency = st.selectbox('Agency:', agencies)

        # Number of passengers
        num_passengers = st.number_input("Number of Passengers", min_value=1, value=1, step=1)

        # Add predict button
        if st.button('Predict Price'):
            # Transform inputs using label encoders
            from_encoded = le_from.transform([from_location])[0]
            to_encoded = le_to.transform([to_location])[0]
            type_encoded = le_type.transform([flight_type])[0]
            agency_encoded = le_agency.transform([agency])[0]

            # Create input array
            input_data = np.array([[from_encoded, to_encoded, agency_encoded, type_encoded, time, distance]])

            # Make prediction
            prediction = model.predict(input_data)[0]
            total_price = prediction * num_passengers

            # Display result
            st.success(f'Predicted Price for {num_passengers} passenger(s): R$ {total_price:.2f}')

if __name__ == '__main__':
    flight_price_prediction_page()