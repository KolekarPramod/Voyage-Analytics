# main_dashboard.py
import streamlit as st

def main():
    st.set_page_config(
        page_title="Welcome to Voyage Analytics",
        page_icon="🚀",
    )
    st.title("Welcome to Voyage Analytics")
    st.header("Integrating MLOps for Predictive and Recommender Systems in Travel Project")

    st.write("Select a model to launch:")

    if st.button("Flight Price Predictor"):
        st.experimental_rerun()  # Clear previous state
        run_flight_price_predictor()

    if st.button("Gender Classification"):
        st.experimental_rerun()  # Clear previous state
        run_gender_classification()

    if st.button("Hotel Recommendation"):
        st.experimental_rerun()  # Clear previous state
        run_hotel_recommendation()

def run_flight_price_predictor():
    st.title("Flight Price Predictor")
    import regression
    regression.main()

def run_gender_classification():
    st.title("Gender Classification")
    import classification
    classification.main()

def run_hotel_recommendation():
    st.title("Hotel Recommendation")
    import recommendation
    recommendation.main()

if __name__ == "__main__":
    main()
