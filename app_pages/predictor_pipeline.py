# app_pages/pipeline_overview.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import pickle
import os
import joblib  


  


def page_pipeline_overview():
    try:
        # Load and inspect encoders and filters
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/encoders_and_filters.pkl', 'rb') as f:
            encoders_and_filters = pickle.load(f)
        
        st.markdown("### 1. Encoders and Filters Pipeline Components")
        st.markdown("**Pipeline Steps:**")

        pipeline_str = "Pipeline(steps=[\n"
        for key in encoders_and_filters.keys():
            pipeline_str += f"    ('{key}'),\n"
        pipeline_str += "])"
        st.code(pipeline_str)

        # Display description of what each step does
        st.markdown("**Pipeline Steps Description:**")
        for step in steps:
            if 'mlb' in step:
               st.write(f"- `{step}`: Multi-label binarization for {step.replace('mlb_', '')}")
            elif 'min_appearances' in step:
               st.write(f"- `{step}`: Frequency threshold filtering")
            elif 'frequent' in step:
               st.write(f"- `{step}`: Selection of frequent items")
            elif 'positions' in step:
               st.write(f"- `{step}`: Position filtering")
            elif 'encoder' in step:
               st.write(f"- `{step}`: Label encoding")
            else:
                st.write(f"- `{step}`: Feature creation")
    
    except Exception as e:
        st.error(f"Error loading pipelines: {str(e)}")
        st.error(f"Full error details: {str(type(e).__name__)}: {str(e)}")

       

      

    '''
        st.title("ML Pipeline Structure")

        st.markdown("**There are 2 ML Pipelines arranged in series.**")
        st.markdown("* The first is responsible for data cleaning and feature engineering.")
    try:
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/encoders_and_filters.pkl', 'rb') as f:
            encoders_and_filters = pickle.load(f)
    
            pipeline_str = "Pipeline(steps=[\n"
            for key in encoders_and_filters.keys():
            
        # Display pipeline components in multiple rows
                pipeline_str += f"    ('{key}', MultiLabelBinarizer()),\n" if 'mlb' in key \
                    else f"    ('{key}', FrequencyThresholdFilter()),\n" if 'min_appearances' in key \
                    else f"    ('{key}', ColumnSelector()),\n" if 'frequent' in key \
                    else f"    ('{key}', PositionFilter()),\n" if 'positions' in key \
                    else f"    ('{key}', LabelEncoder()),\n" if 'encoder' in key \
                    else f"    ('{key}', FeatureCreator()),\n"
            pipeline_str = pipeline_str.rstrip(',\n') + "\n])"
       
        st.code(pipeline_str)

        st.markdown("---")
        st.markdown("**Second Pipeline: Feature Engineering**")
        
            
        # Create pipeline string for second pipeline
        feature_pipeline_str = """Pipeline(steps=["
        components = {
            'budget_imputer': 'SimpleImputer(strategy="median")',
            'runtime_imputer': 'SimpleImputer(strategy="median")',
            'release_date_transformer': 'DateFeatureExtractor()',
            'budget_scaler': 'StandardScaler()',
            'runtime_scaler': 'StandardScaler()',
            'target_encoder': 'LogTransformer()'
        ])
    
        
        st.code(feature_pipeline_str)
       
        st.info(
            f"* This pipeline uses a Random Forest model optimized through GridSearchCV to predict movie revenues. \n"
            f"* The model achieved an R² score of 0.15, showing the challenge of predicting exact movie revenues. \n"
            f"* Budget is the most influential feature, accounting for 46.7% of prediction weight. \n"
            f"* Genre factors like Comedy (4.6%), Drama (4.4%), and Thriller (4.2%) also play significant roles. \n"
            f"* The model performs with an RMSE of $1.06 and MAE of $0.89 on logged, scaled revenue values."
        )
    '''
          
    



  
    
    



   
  

