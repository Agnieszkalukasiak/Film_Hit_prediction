import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
import seaborn as sns 
import pickle


def load_data():
    """Load all necessary models and data"""
    try:
        # Load the model
        model = joblib.load('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/models/film_revenue_model_Random Forest_20250115.joblib')
        
        # Load transformation data
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/full_transformation_data.pkl', 'rb') as f:
            transform_data = pickle.load(f)

        
        # Load feature scaler
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/feature_scaler.pkl', 'rb') as f:
            feature_scaler = pickle.load(f)
            
        # Load top revenue data
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_actors.pkl', 'rb') as f:
            top_actors = pickle.load(f)
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_directors.pkl', 'rb') as f:
            top_directors = pickle.load(f)
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_writers.pkl', 'rb') as f:
            top_writers = pickle.load(f)
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_producers.pkl', 'rb') as f:
            top_producers = pickle.load(f)
            
        return (model, transform_data, feature_scaler, predict_movie_revenue, 
                top_actors, top_directors, top_writers, top_producers)
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def page_predictor_body():
    st.title('Movie Revenue Predictor 🎬')
    
    try:
        # Load all required data
        data = load_data()
        if data is None:
            return
            
        (model, transform_data, feature_scaler,
         top_actors, top_directors, top_writers, top_producers) = data
        
        # Create form
        st.write("Enter movie details:")
        
        # Basic movie info
        col1, col2 = st.columns(2) 

        with col1:
            budget = st.number_input('Budget ($)', min_value=0, value=150000000)
            runtime = st.number_input('Runtime (minutes)', min_value=0, value=120)
            
        with col2:
            # Get languages from encoder
            language = st.selectbox('Language', transform_data['encoders_and_filters']['language_encoder'].classes_)
            # Get countries
            production_country = st.selectbox('Production Country', 
                                           transform_data['encoders_and_filters']['frequent_countries']+ ['Other'])
            if production_country == 'Other':
                production_country = st.text_input('Enter Production Country Name')


        # Genre selection (multiple)
        genres = transform_data['genre_columns']
        selected_genres = st.multiselect('Select Genres', genres)
        
        # Production company
        production_company = st.selectbox('Production Company', 
                                        transform_data['encoders_and_filters']['frequent_companies']+ ['Other'])
        if production_company == 'Other':
            production_company = st.text_input('Enter Production Company Name')
        
        # Cast and Crew
        st.subheader('Cast and Crew')
        col3, col4 = st.columns(2)
        
        with col3:
            actor1 = st.selectbox('Lead Actor', top_actors['columns']+ ['Other'])

            if actor1 == 'Other':
                actor1 = st.text_input('Enter Lead Actor Name')

            actor2 = st.selectbox('Supporting Actor', top_actors['columns']+ ['Other'])
            if actor2 == 'Other':
                actor2 = st.text_input('Enter Supporting Actor Name')

            director = st.selectbox('Director', top_directors['columns']+ ['Other'])
            if director == 'Other':
                director = st.text_input('Enter Director Name')
            
        with col4:
            writer = st.selectbox('Writer', top_writers['columns']+ ['Other'])
            if writer == 'Other':
                writer = st.text_input('Enter Writer Name')

            producer = st.selectbox('Producer', top_producers['columns']+ ['Other'])
            if producer == 'Other':
                producer = st.text_input('Enter Producer Name')
        
        if st.button("Predict Revenue"):
            try:
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
                        st.metric(label="Profit/Loss", value=f"${prediction['profit']:,.2f}", 
                                delta=f"${prediction['profit']:,.2f}")
                    with col3:
                        st.metric(label="ROI", value=f"{prediction['roi']:.1f}%", 
                                delta=f"{prediction['roi']:.1f}%")
                        
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
                
    except Exception as e:
        st.error(f"Error loading required data: {str(e)}")
        st.info("Please ensure all required files are available.")



page_predictor_body()
