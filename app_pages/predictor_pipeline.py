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
    col1, col2, col3 = st.columns(3)
    col1.metric("Initial Rows", raw_data_metrics["Initial Rows"])
    col2.metric("Missing Values", raw_data_metrics["Missing Values"])
    col3.metric("Features", raw_data_metrics["Features"])

    # Data Preparation
    st.header("2. Data Preparation")
    with st.expander("See Details"):
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
    with st.expander("See Details"):
        st.write("""
        ### Model Selection:
        1. **Algorithms Tested**
           - Linear Regression
           - Random Forest
           - XGBoost
        
        2. **Best Model**: Random Forest
           - Best performance on validation set
           - Good balance of interpretability
        
        3. **Hyperparameter Tuning**
           - Used GridSearchCV
           - Optimized for R² score
        """)

    # Evaluation
    st.header("4. Model Evaluation")
    with st.expander("See Details"):
        eval_metrics = {
            "R² Score": "0.15",
            "RMSE": "$1.06",
            "MAE": "$0.89"
        }
        
        col1, col2, col3 = st.columns(3)
        col1.metric("R² Score", eval_metrics["R² Score"])
        col2.metric("RMSE", eval_metrics["RMSE"])
        col3.metric("MAE", eval_metrics["MAE"])

        st.write("""
        ### Key Findings:
        - Budget is the strongest predictor (46.7% importance)
        - Genre factors like Comedy (4.6%) and Drama (4.4%) are also significant
        - Model performs best on mainstream budget ranges
        """)

    # Pipeline Visualization
    st.header("5. Complete Pipeline Flow")
    pipeline_diagram = """
    graph TD
        A[Raw TMDB Data] --> B[Data Cleaning]
        B --> C[Feature Engineering]
        C --> D[Model Training]
        D --> E[Model Evaluation]
        E --> F[Model Deployment]
        F --> G[Prediction Interface]
    """
    st.mermaid(pipeline_diagram)