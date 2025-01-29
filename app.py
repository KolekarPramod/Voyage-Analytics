import streamlit as st
from gender_classification import gender_classification_page
from hotel_recommendation import hotel_recommendation_page
from flight_price_prediction import flight_price_prediction_page

def main():
    st.set_page_config(page_title="Voyage Analytics", page_icon=":earth_americas:")

    st.sidebar.title('Navigation')
    selection = st.sidebar.radio("Go to", ["Home", "Flight Price Prediction", "Gender Classification", "Hotel Recommendation"])

    if selection == "Home":
        st.title('Voyage Analytics: Integrating MLOps for Predictive and Recommender Systems in Travel')
        st.header('Productionization of ML Systems')
        st.write("""
        ### Project Description
        This project leverages data analytics and machine learning to revolutionize travel experiences. Using datasets on users, flights, and hotels, we aim to enhance predictive capabilities and deploy sophisticated machine learning models, all while mastering MLOps through hands-on application.
        """)
        st.write("""
        ### Business Context
        In the travel and tourism industry, the combination of data analytics and machine learning provides an opportunity to transform how travel experiences are curated and delivered. This project involves:
        - **Predictive Modeling:** Enhancing decision-making related to travel through predictive analytics.
        - **MLOps Mastery:** Implementing end-to-end machine learning operations to ensure seamless deployment and maintenance of models.
        """)
    elif selection == "Flight Price Prediction":
        flight_price_prediction_page()
    elif selection == "Gender Classification":
        gender_classification_page()
    elif selection == "Hotel Recommendation":
        hotel_recommendation_page()

if __name__ == '__main__':
    main()