import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
import seaborn as sns 

def page_predictor_pipeline_body():
    st.title("Movie Revenue Prediction Pipeline")

    # load needed files
    try:
        engineered_path = "outputs/datasets/engineered/"
        model_path = "outputs/models/"
    
    #Load components
        le_language = joblib.load(os.path.join(engineered_path, 'language_encoder.joblib'))
        scaler = joblib.load(os.path.join(engineered_path, 'budget_scaler.joblib'))
        scaler_y = joblib.load(os.path.join(engineered_path, 'revenue_scaler.joblib'))
        model = joblib.load(os.path.join(model_path, 'movie_revenue_predictor.joblib'))


    # display pipeline training summary conclusions
        st.info(
            f"* The pipeline was tuned aiming at least 0.80 Recall on 'Yes Churn' class, "
            f"since we are interested in this project in detecting a potential churner. \n"
            f"* The pipeline performance on train and test set is 0.90 and 0.85, respectively."
        )

        st.write("### ML Pipeline Steps")

    # Sample input
        st.header("1. Enter Movie Details")

        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input("Movie Budget ($)", 
                                   min_value=100000, 
                                   max_value=500000000, 
                                   value=1000000,
                                   step=100000,
                                   format="%d")
   
            language = st.selectbox("Movie Language", 
                                  options=le_language.classes_,
                                  index=list(le_language.classes_).index('en'))
        
        with col2:
            all_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
                         'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                         'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
                         'TV Movie', 'Thriller', 'War', 'Western', 'Foreign']
            
            genres = st.multiselect("Select Genres", 
                                  options=all_genres,
                                  default=['Action'])
        #Feature processing
        if st.button("Predict Revenue"):
            st.header("2. Feature Processing")
            
            # Budget processing
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Original Budget:")
                st.write(f"${budget:,.2f}")
            
            budget_logged = np.log1p(budget)
            with col2:
                st.write("Log Transformed:")
                st.write(f"{budget_logged:.2f}")
            
            budget_scaled = scaler.transform([[budget_logged]])[0][0]
            with col3:
                st.write("Scaled Budget:")
                st.write(f"{budget_scaled:.2f}")
       


