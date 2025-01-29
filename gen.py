import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Name Gender Predictor",
    page_icon="👤",
    layout="centered"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stTextInput > div > div > input {
        font-size: 20px;
        padding: 15px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
        text-align: center;
        background-color: #f0f2f6;
    }
    .big-font {
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with st.spinner('Loading model... (this might take a minute)'):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model

def train_classifier(df):
    # Prepare data
    df = df.dropna(subset=['name', 'gender'])
    df['gender_label'] = (df['gender'] == 'female').astype(int)
    
    # Get embeddings
    model = load_model()
    X = model.encode(df['name'].tolist(), show_progress_bar=False)
    y = df['gender_label']
    
    # Train classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X, y)
    
    return model, clf

def predict_gender(name, embedding_model, classifier):
    embedding = embedding_model.encode([name], show_progress_bar=False)
    prediction = classifier.predict(embedding)[0]
    probability = classifier.predict_proba(embedding)[0]
    
    gender = 'female' if prediction == 1 else 'male'
    confidence = probability[1] if gender == 'female' else probability[0]
    
    return {
        'name': name,
        'predicted_gender': gender,
        'confidence': round(confidence * 100, 2)
    }

def create_gauge_chart(confidence, gender):
    color = '#FF69B4' if gender == 'female' else '#4169E1'
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence,
        title = {'text': "Confidence Score", 'font': {'size': 24}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': '#f2f2f2'},
                {'range': [33, 66], 'color': '#e6e6e6'},
                {'range': [66, 100], 'color': '#d9d9d9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=50, b=10),
        font={'size': 16}
    )
    
    return fig

def main():
    st.title("🧑👩 Name Gender Predictor")
    st.markdown("### Enter a name to predict its gender")
    
    # Initialize session state for model
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.classifier = None
    
    # Load default dataset and train model if not already done
    if st.session_state.model is None:
        try:
            df = pd.read_csv('pramod_user.csv')
            st.session_state.model, st.session_state.classifier = train_classifier(df)
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return
    
    # Input field for name
    name = st.text_input("", placeholder="Enter a name here...", key="name_input")
    
    # Process input when a name is entered
    if name:
        name = name.strip()
        if len(name) > 0:
            with st.spinner('Analyzing...'):
                result = predict_gender(name, st.session_state.model, st.session_state.classifier)
                
                # Display prediction
                gender_color = '#FF69B4' if result['predicted_gender'] == 'female' else '#4169E1'
                st.markdown(f"""
                <div class='prediction-box'>
                    <div class='big-font'>Prediction for '{result['name']}'</div>
                    <div style='color: {gender_color}; font-size: 36px; font-weight: bold;'>
                        {result['predicted_gender'].upper()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Display confidence gauge
                fig = create_gauge_chart(result['confidence'], result['predicted_gender'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Add interpretation
                confidence_level = "High" if result['confidence'] >= 90 else "Medium" if result['confidence'] >= 70 else "Low"
                st.markdown(f"""
                    #### Interpretation
                    - **Confidence Level**: {confidence_level}
                    - **Probability**: {result['confidence']}%
                """)

if __name__ == "__main__":
    main()