import streamlit as st
from regression import main as regression_main
from recommendation import main as recommendation_main
from classification import main as classification_main

def main():
    st.set_page_config(
        page_title="Voyage Analytics Home",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Voyage Analytics: Integrating MLOps for Predictive and Recommender Systems in Travel")
    st.subheader("Productionization of ML Systems")

    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Home", "Flight Price Predictor", "Hotel Recommendation", "Gender Classification"])

    if page == "Home":
        st.write("""
        Welcome to Voyage Analytics! This platform demonstrates the integration of MLOps techniques for predictive and recommender systems in the travel domain. You can explore different functionalities by navigating through the sidebar options:
        
        - **Flight Price Predictor**: Predict flight prices based on various factors.
        - **Hotel Recommendation**: Get hotel recommendations based on user preferences.
        - **Gender Classification**: Classify the gender of a user based on input details.

        Choose a functionality from the sidebar to get started!
        """)

    elif page == "Flight Price Predictor":
        regression_main()

    elif page == "Hotel Recommendation":
        recommendation_main()

    elif page == "Gender Classification":
        classification_main()

if __name__ == '__main__':
    main()
