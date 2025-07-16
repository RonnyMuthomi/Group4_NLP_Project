import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os


# === Sticky header and styles ===
st.markdown("""
    <style>
        .main .block-container {
            padding-top: 0rem;
        }

        .sticky-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            background-color: #1f77b4;
            color: white;
            padding: 15px 30px;
            display: flex;
            align-items: center;
            z-index: 999;
            box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.2);
        }

        .sticky-header img {
            height: 40px;
            margin-right: 15px;
        }

        .sticky-header h1 {
            font-size: 24px;
            margin: 0;
        }

        .header-spacer {
            height: 80px;
        }

        .fake-sidebar {
            background-color: #ffa94d;
            padding: 20px;
            border-radius: 10px;
            height: 100%;
        }

        .card-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            flex-wrap: wrap;
        }

        .card {
            background-color: white;
            color: black;
            padding: 20px;
            width: 90%;
            max-width: 400px;
            border-radius: 12px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            text-align: center;
            transition: transform 0.2s;
            margin: 10px;
        }

        .card:hover {
            transform: translateY(-5px);
            background-color: #f1f1f1;
        }

        .card h3 {
            color: #1f77b4;
        }
    </style>

    <div class="sticky-header">
        <img src="https://upload.wikimedia.org/wikipedia/commons/f/fa/Apple_logo_black.svg" alt="Logo">
        <h1>Apple Twitter Sentiment Classification</h1>
    </div>

    <div class="header-spacer"></div>
""", unsafe_allow_html=True)




# Load your trained pipeline
pipeline = joblib.load("../models/random_forest_model.pkl")

# Set page config
st.set_page_config(page_title="Tweet Sentiment Analyzer", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    /* Main page background */
    .main {
        background-color: #f5f7fa;
    }

    /* Header styling */
    .big-title {
        background-color: #004488;
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 32px;
        margin-bottom: 20px;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffa726;
    }

    /* Prediction card */
    .result-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        text-align: center;
        font-size: 22px;
        font-weight: bold;
    }

    .footer {
        text-align: center;
        font-size: 14px;
        color: gray;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<div class='big-title'>Tweet Sentiment Analyzer</div>", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("Analyze Tweet")
    tweet_input = st.text_area("Type a tweet below:", height=150)
    st.markdown("""
    **About Machine Learning**

    Machine Learning (ML) allows computers to learn from data and make predictions or decisions without being explicitly programmed. In this app, ML helps us detect **emotions** from tweet content using a trained model.

    """)
    predict_button = st.button("🔮 Predict Sentiment")

# --- Predict ---
if predict_button:
    if tweet_input.strip() == "":
        st.warning("Please enter a tweet to analyze.")
    else:
        
        input_df = pd.DataFrame({'processed_tweet': [tweet_input]})
        prediction = pipeline.predict(input_df)[0]

        sentiment_colors = {
            "Positive emotion": "#4CAF50",
            "Negative emotion": "#F44336",
            "No emotion toward brand or product": "#9E9E9E",
            "I can't tell": "#FF9800"
        }

        # Show as a card
        st.markdown(f"""
            <div class="result-card" style="color:{sentiment_colors.get(prediction, 'black')};">
                🎯 Predicted Sentiment: <br> {prediction}
            </div>
        """, unsafe_allow_html=True)

# --- About the Project Section ---
st.subheader("📘 About the Project")
st.write("""
This project leverages Natural Language Processing (NLP) to classify the **sentiment of tweets** into categories such as:
- Positive emotion  
- Negative emotion  
- No emotion toward product  
- I can't tell  

It uses **TF-IDF vectorization** to convert text into numeric form, followed by a **trained machine learning model** that can understand and predict the sentiment of unseen tweets.
""")

# --- Optional Insight / Chart ---
st.subheader("Sample Sentiment Distribution")
sample_labels = ['Positive emotion', 'Negative emotion', 'No emotion', "I can't tell"]
sample_counts = [3000, 700, 5000, 300]

fig, ax = plt.subplots()
bars = ax.bar(sample_labels, sample_counts, color=['#4CAF50', '#F44336', '#9E9E9E', '#FF9800'])
ax.set_ylabel("Tweet Count")
ax.set_title("Distribution of Sentiment Classes (Sample Data)")
st.pyplot(fig)

# --- Footer ---
st.markdown("<div class='footer'>👨‍💻 Developed by Group4 members | NLP Sentiment Analysis Project 2025</div>", unsafe_allow_html=True)