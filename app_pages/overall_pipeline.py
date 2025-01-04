import streamlit as st

def page_predictor_pipeline_body():
    st.write("### ML Pipeline: Movie Revenue Prediction")

    with st.expander("1. Data Collection"):
        st.write("- Dataset: Movies from TMDB API")
        st.write("- Features: title, budget, genres, language, etc.")
        st.write("- Target: Revenue")
        st.image("figures/data_collection.png", caption="Raw Data Sample")

    with st.expander("2. Data Cleaning"):
        st.write("- Removed duplicates")
        st.write("- Handled missing values")
        st.write("- Filtered invalid entries")
        st.image("figures/data_cleaning.png", caption="Data Cleaning Process")

    with st.expander("3. Film Success Study"):
        st.write("- ROI Analysis")
        st.write("- Budget-Revenue Relationship")
        st.write("- Genre Impact")
        st.image("figures/success_study.png", caption="Success Metrics")

    with st.expander("4. Feature Engineering"):
        st.write("- Budget scaling")
        st.write("- Language encoding")
        st.write("- Genre one-hot encoding")
        st.image("figures/feature_engineering.png", caption="Engineered Features")

    with st.expander("5. Modeling & Evaluation"):
        st.write("- Model: Random Forest Regressor")
        st.write("- Metrics: R², RMSE, MAE")
        st.write("- Feature Importance")
        st.image("figures/model_evaluation.png", caption="Model Performance")