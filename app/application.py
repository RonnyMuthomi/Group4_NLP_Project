import streamlit as st
import pandas as pd
import joblib
import os

# === Load the model ===
@st.cache_resource
def load_model():
    model_path = os.path.join("../models/random_forest_model.pkl")
    return joblib.load(model_path)

model = load_model()

# === Predict function ===
def predict_sentiment(tweet):
    input_df = pd.DataFrame({'processed_tweet': [tweet]})
    sentiment = model.predict(input_df)[0]
    return {
        "tweet": tweet,
        "sentiment": sentiment
    }

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

