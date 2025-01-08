# app_pages/pipeline_overview.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def page_pipeline_overview():
    st.title("Data Science Pipeline Overview")


# display pipeline training summary conclusions
    
    st.info(
        f"* This pipeline uses a Random Forest model optimized through GridSearchCV to predict movie revenues. \n"
        f"* The model achieved an R² score of 0.15, showing the challenge of predicting exact movie revenues. \n"
        f"* Budget is the most influential feature, accounting for 46.7% of prediction weight. \n"
        f"* Genre factors like Comedy (4.6%), Drama (4.4%), and Thriller (4.2%) also play significant roles. \n"
        f"* The model performs with an RMSE of $1.06 and MAE of $0.89 on logged, scaled revenue values."
        )

    # Data Understanding
    st.header("Data Loading and Understanding")
    st.write("""
    - **Source Data**: TMDB Movie Dataset
    - **Initial Features**: 23 columns including budget, revenue, genres
    - **Initial Rows**: 10,000 movie records
    - **Time Period**: 1990-2023
    """)
        
    # Show sample of raw data
    st.write("**Dataset Metrics:**")
    st.write("* Initial Rows: 10,000")
    st.write("* Missing Values: 15%")
    st.write("* Features: 23")

    # Data Preparation
    st.header("2. Data Preparation")
    st.write("""
    ### Cleaning Steps:
    1. **Missing Values**
        - Removed rows with missing budgets
        - Imputed missing languages with 'en'
        
    2. **Outlier Handling**
        - Removed movies with $0 budget
        - Filtered extreme budget outliers
        
    3. **Feature Engineering**
        - Log transformed budget and revenue
        - One-hot encoded genres
        - Label encoded languages
    """)

    # Modeling
    st.header("3. Model Development")
    st.write("""
    ### Model Selection:
    1. **Algorithms Tested**
       - Linear Regression
       - Ridge
       - Lasso
       - Random Forest
       - Gradient Boosting
        
    2. **Hyperparameter Optimization**
       - GridSearchCV with cross-validation
       - Random Forest params: depths 10-30, estimators 100-500
       - Gradient Boosting params: learning rates 0.01-0.3
    
    3. ** Best Model: Random Forest Regressor
    **Parameters**:
       - n_estimators: [100, 200, 500]
       - learning_rate: [0.01, 0.1, 0.3]
       - max_depth: [3, 5, 7]
       - subsample: [0.8, 0.9, 1.0]
    """)

    # Evaluation
    st.header("4. Model Evaluation")
    st.write("""
    ### Evaluation Metrics:
    * "R² Score": "0.15",
    * "RMSE": "$1.06",
    * "MAE": "$0.89"
    
    ### Key Findings:
    - Budget is the strongest predictor (46.7% importance)
    - Genre factors like Comedy (4.6%) and Drama (4.4%) are also significant
    - Model performs best on mainstream budget ranges
    """)
    

    # Pipeline Visualization
    st.header("5. Complete Pipeline Flow")
    st.write("""
    **Pipeline Steps:**
    1. Raw TMDB Data ➡️
    2. Data Cleaning ➡️
    3. Data Exploration/Correlation study ➡️
    3. Feature Engineering ➡️
    4. Model Training ➡️
    5. Model Evaluation ➡️
    6. Model Deployment ➡️
    7. Prediction Interface
    """)