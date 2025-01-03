import streamlit as st
import pandas as pd
import numpy as np
import joblib

def predict_revenue_page():
    st.write("### Movie Revenue Predictor")
    st.info("Predict movie revenue and profitability based on budget, language and genre.")
    
    # Input widgets
    col1, col2 = st.columns(2)
    
    with col1:
        budget = st.number_input("Budget ($)", min_value=100000, max_value=500000000, value=10000000)
        language = st.selectbox("Language", ['en', 'fr', 'es', 'de', 'it', 'ja', 'zh', 'hi', 'ko', 'sv'])
        
    with col2:
        genres = st.multiselect(
            "Genres",
            ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary', 
             'Drama', 'Family', 'Fantasy', 'History', 'Horror', 'Music', 'Mystery',
             'Romance', 'Science Fiction', 'TV Movie', 'Thriller', 'War', 'Western', 'Foreign'],
            default=['Drama']
        )

    if st.button("Predict Revenue"):
        result = predict_movie_metrix(budget, language, genres)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Budget", f"${result['budget']:,.2f}")
        with col2:
            st.metric("Predicted Revenue", f"${result['predicted_revenue']:,.2f}")
        with col3:
            st.metric("ROI", f"{result['roi']:.1f}%")

        if result['is_profitable']:
            st.success(f"PROFIT: ${result['profit_amount']:,.2f}")
        else:
            st.error(f"LOSS: ${result['loss_amount']:,.2f}")

if __name__ == "__main__":
    predict_revenue_page()
