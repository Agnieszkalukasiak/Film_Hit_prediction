import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
import seaborn as sns 


def load_encoders():
    with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/encoders_and_filters.pkl', 'rb') as f:
        return pickle.load(f)

def load_model():
    return joblib.load('/workspace/Film_Hit_prediction/outputs/models/movie_revenue_predictor.joblib')

def page_predictor_body():
    st.title('Movie Revenue Predictor')
    
    # Load encoders
    encoders = load_encoders()
    
    # Create form
    with st.form("movie_prediction_form"):
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
                 'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                 'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
                 'TV Movie', 'Thriller', 'War', 'Western']
        selected_genres = st.multiselect('Select Genres', genres)
        
        # Production company
        companies = sorted(encoders['company_encoder'].classes_)
        production_company = st.selectbox('Production Company', companies)
        
        # Cast and Crew
        st.subheader('Cast and Crew')
        col3, col4 = st.columns(2)
        
        with col3:
            actor1 = st.text_input('Actor 1')
            actor2 = st.text_input('Actor 2')
            director = st.text_input('Director')
            
        with col4:
            writer = st.text_input('Writer')
            producer = st.text_input('Producer')
        
        submitted = st.form_submit_button("Predict Revenue")
        
        if submitted:
            try:
                # Prepare prediction matrix
                input_data = {
                    'budget': budget,
                    'runtime': runtime,
                    'genres': selected_genres,
                    'language': language,
                    'production_company': production_company,
                    'production_country': production_country,
                    'actor1': actor1,
                    'actor2': actor2,
                    'crew_director': director,
                    'crew_writer': writer,
                    'crew_producer': producer
                }
                
                # Create prediction matrix
                pred_matrix = prepare_prediction_matrix(**input_data)
                
                if pred_matrix is not None:
                    # Load and use model
                    model = load_model()
                    predicted_revenue = model.predict(pred_matrix)[0]
                    
                    # Display results
                    st.success('Prediction successful!')
                    
                    st.metric(
                        label="Predicted Revenue",
                        value=f"${predicted_revenue:,.2f}"
                    )
                    
                    # Calculate and display ROI
                    roi = ((predicted_revenue - budget) / budget) * 100
                    st.metric(
                        label="Predicted ROI",
                        value=f"{roi:.1f}%",
                        delta=f"{roi:.1f}%" if roi > 0 else f"{roi:.1f}%"
                    )
                    
                    # Display movie details
                    st.subheader('Movie Details')
                    details = {
                        'Budget': f"${budget:,.2f}",
                        'Runtime': f"{runtime} minutes",
                        'Genres': ', '.join(selected_genres),
                        'Language': language,
                        'Production Company': production_company,
                        'Production Country': production_country,
                        'Cast': f"{actor1}, {actor2}",
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
    main()