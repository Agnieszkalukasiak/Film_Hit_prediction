# app_pages/pipeline_overview.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBClassifier
import pickle

def load_pickle(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"File not found: {file_path}")
            return None
        return joblib.load(file_path)
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

def page_pipeline_overview():

    st.title("ML Pipeline Structure")
    
    version = 'v1'
    
    # Load both pipelines
    cleaning_pipeline = load_pickle(f"/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/{version}/cleaning_pipeline.pkl")
    modeling_pipeline = load_pickle(f"/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/models/{version}/movie_feature_engineering_pipeline.pkl")
    
    # Display pipelines side by side with arrow
    col1, arrow, col2 = st.columns([2, 1, 2])
    
    with col1:
        st.subheader("Data Cleaning Pipeline")
        st.code(str(cleaning_pipeline))

    with arrow:
        st.markdown("<div style='text-align: center; font-size: 24px; padding-top: 50px;'>➔</div>", 
                   unsafe_allow_html=True)

    with col2:
        st.subheader("Feature Engineering Pipeline")
        st.code(str(modeling_pipeline))

    
    st.info(
    f"* This pipeline uses a Random Forest model optimized through GridSearchCV to predict movie revenues. \n"
    f"* The model achieved an R² score of 0.15, showing the challenge of predicting exact movie revenues. \n"
    f"* Budget is the most influential feature, accounting for 46.7% of prediction weight. \n"
    f"* Genre factors like Comedy (4.6%), Drama (4.4%), and Thriller (4.2%) also play significant roles. \n"
    f"* The model performs with an RMSE of $1.06 and MAE of $0.89 on logged, scaled revenue values."
    )


if __name__ == "__main__":
    show_pipelines()

   
  

