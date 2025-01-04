import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def page_predictor_pipeline_body():

    # load needed files
    engineered_path = "/workspace/Film_Hit_prediction/outputs/datasets/engineered/"
    model_path = "/workspace/Film_Hit_prediction/outputs/models/"
    
    le_language = joblib.load(engineered_path + 'language_encoder.joblib')
    scaler = joblib.load(engineered_path + 'budget_scaler.joblib')
    scaler_y = joblib.load(engineered_path + 'revenue_scaler.joblib')
    model = joblib.load(model_path + 'movie_revenue_predictor.joblib')

    st.write("### ML Pipeline: Predict Prospect Churn")

    # display pipeline training summary conclusions
    st.info(
        f"* The pipeline was tuned aiming at least 0.80 Recall on 'Yes Churn' class, "
        f"since we are interested in this project in detecting a potential churner. \n"
        f"* The pipeline performance on train and test set is 0.90 and 0.85, respectively."
    )

def page_predictor_pipeline_body():
   engineered_path = "/workspace/Film_Hit_prediction/outputs/datasets/engineered/"
   model_path = "/workspace/Film_Hit_prediction/outputs/models/"

   # Load components
   le_language = joblib.load(engineered_path + 'language_encoder.joblib')
   scaler = joblib.load(engineered_path + 'budget_scaler.joblib')
   scaler_y = joblib.load(engineered_path + 'revenue_scaler.joblib')
   model = joblib.load(model_path + 'movie_revenue_predictor.joblib')

   st.write("### ML Pipeline Steps")

   # Sample input
   sample = {
       'budget': 1000000,
       'language': 'en',
       'genres': ['Action']
   }

   st.write("#### 1. Raw Input")
   st.write(sample)

   # Budget transformation
   budget_logged = np.log1p(sample['budget']) 
   budget_scaled = scaler.transform([[budget_logged]])[0][0]

   st.write("#### 2. Budget Processing")
   st.write(f"Log transform: {budget_logged:.2f}")
