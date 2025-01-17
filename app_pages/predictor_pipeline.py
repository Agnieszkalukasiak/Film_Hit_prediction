# app_pages/pipeline_overview.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

def load_pkl_file(filepath):
    try:
        print(f"Loading file from: {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        raise

def page_pipeline_overview():
    st.title("Data Science Pipeline Overview")
    version = 'v1'

    try:
        filepath = '/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/encoders_and_filters.pkl'
        print(f"Absolute path: {os.path.abspath(filepath)}")

        encoders_and_filters = load_pkl_file(filepath)

        # Display the loaded data
        st.write("### Encoders and Filters Data")

    # Display the keys if it is a dictionary-like object
        if isinstance(encoders_and_filters, dict):
            st.write("List of keys in the data:")
            for key in encoders_and_filters.keys():
                st.write(f"- {key}")
    except Exception as e:
        st.error(f"Error loading or processing data: {str(e)}")
        st.write("Please check the file path and ensure the pickle file exists.")
   

if __name__ == "__main__":
    page_pipeline_overview()

# display pipeline training summary conclusions
    
st.info(
    f"* This pipeline uses a Random Forest model optimized through GridSearchCV to predict movie revenues. \n"
    f"* The model achieved an R² score of 0.15, showing the challenge of predicting exact movie revenues. \n"
    f"* Budget is the most influential feature, accounting for 46.7% of prediction weight. \n"
    f"* Genre factors like Comedy (4.6%), Drama (4.4%), and Thriller (4.2%) also play significant roles. \n"
    f"* The model performs with an RMSE of $1.06 and MAE of $0.89 on logged, scaled revenue values."
    )
