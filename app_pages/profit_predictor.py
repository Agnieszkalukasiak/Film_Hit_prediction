import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
import seaborn as sns 
import pickle


def load_encoders():
    with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/encoders_and_filters.pkl', 'rb') as f:
        return pickle.load(f)
    # Debug information
    st.write("Encoders loaded:", encoders.keys())

def load_model():
    return joblib.load('/workspace/Film_Hit_prediction/outputs/models/film_revenue_predictor.joblib')


def page_predictor_body():
    st.title('Movie Revenue Predictor 🎬')
    
    # Load encoders
    encoders = load_encoders()
    
    # Create form
    
    st.write("Enter movie details:")
    # Basic movie info
    col1, col2 = st.columns(2)
        
    with col1:
        budget = st.number_input('Budget ($)', min_value=0, value=1000000)
        runtime = st.number_input('Runtime (minutes)', min_value=0, value=120)
            
    with col2:
        # Get unique languages from encoder
        languages = sorted(encoders['language_encoder'].classes_)
        language = st.selectbox('Language', languages)
            
        # Get unique countries from encoder
        countries = sorted(encoders['country_encoder'].classes_)
        production_country = st.selectbox('Production Country', countries)
        
    # Genre selection (multiple)
    genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Family', 'Fantasy', 'Horror',
                'Mystery', 'Romance', 'Science Fiction', 'Thriller']
    selected_genres = st.multiselect('Select Genres', genres)
        
    # Production company
    companies = sorted(encoders['company_encoder'].classes_)
    production_company = st.selectbox('Production Company', companies)
        
    # Cast and Crew
    st.subheader('Cast and Crew')
    col3, col4 = st.columns(2)
        
    with col3:
        actor1 = st.text_input('Lead Actor')
        actor2 = st.text_input('Supporting Actor')
        director = st.text_input('Director')
            
    with col4:
        writer = st.text_input('Writer')
        producer = st.text_input('Producer')
        
      
        
    if st.button("Predict Revenue"):
        try:
            # Use the predict_movie_revenue function directly
            prediction = predict_movie_revenue(
                budget=budget,
                runtime=runtime,
                genres=selected_genres,
                language=language,
                production_company=production_company,
                production_country=production_country,
                actor1=actor1,
                actor2=actor2,
                crew_director=director,
                crew_writer=writer,
                crew_producer=producer
            )
                
            if prediction:
                # Display results in columns
                st.success('Prediction successful! 🎯')
                    
                col1, col2, col3 = st.columns(3)
                    
                with col1:
                    st.metric(label="Predicted Revenue", value=f"${prediction['revenue']:,.2f}")
                with col2:
                    st.metric(label="Profit/Loss", value=f"${prediction['profit']:,.2f}", delta=f"${prediction['profit']:,.2f}")
                with col3:
                    st.metric(label="ROI", value=f"{prediction['roi']:.1f}%", delta=f"{prediction['roi']:.1f}%")
                if prediction['is_profitable']:
                    st.success("🎉 This movie is predicted to be profitable!")
                else:
                    st.warning("⚠️ This movie is predicted to lose money.")
                with st.expander("See detailed analysis", expanded=True):
                    st.subheader('Movie Details')
                    details = {
                        'Budget': f"${budget:,.2f}",
                        'Expected Revenue': f"${prediction['revenue']:,.2f}",
                        'Profit/Loss': f"${prediction['profit']:,.2f}",
                        'Return on Investment': f"{prediction['roi']:.1f}%",
                        'Runtime': f"{runtime} minutes",
                        'Genres': ', '.join(selected_genres),
                        'Language': language,
                        'Production Company': production_company,
                        'Production Country': production_country,
                        'Lead Actor': actor1,
                        'Supporting Actor': actor2,
                        'Director': director,
                        'Writer': writer,
                        'Producer': producer
                    }
                    for key, value in details.items():
                        st.text(f"{key}: {value}")
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please check all inputs and try again.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Movie Revenue Predictor",
        page_icon="🎬",
        layout="wide"
    )
    page_predictor_body()


