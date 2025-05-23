import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
import seaborn as sns 
import pickle
import traceback
import yaml
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import subprocess

def ensure_lfs_files():
    subprocess.run(["git", "lfs", "pull"])

class DummyEncoder:
    def __init__(self):
        self.classes_ = ['en']
    def transform(self, x):
        return [0]

def get_default_values():
    transform_data = {'encoders_and_filters': {'language_encoder': DummyEncoder()}}
    transform_data.update({
        'all_features': ['budget', 'runtime', 'popularity'],
        'numeric_cols': ['budget', 'runtime'],
        'top_actors': {'columns': ['Tom Cruise']},
        'top_directors': {'columns': ['Spielberg']},
        'top_writers': {'columns': ['Sorkin']},
        'top_producers': {'columns': ['Bruckheimer']}
    })
    return transform_data

def load_data():
    try:
        import sys
        print("Python imports:")
        import sklearn
        print(f"sklearn version: {sklearn.__version__}")
        print(f"sklearn path: {sklearn.__path__}")
        
        print("\nPickle header check:")
        path = 'jupyter_notebooks/outputs/engineered/feature_scaler.pkl'
        with open(path, 'rb') as f:
            header = f.read(50)
            print(f"Hex: {header.hex()}")
            print(f"ASCII: {header}")

        print("Loading models and data...")
        model = joblib.load('jupyter_notebooks/outputs/models/film_revenue_model_Random Forest_20250127.joblib')
        transform_data = get_default_values()

        try:
            with open('jupyter_notebooks/outputs/engineered/feature_scaler.pkl', 'rb') as f:
                transform_data['feature_scaler'] = pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load feature scaler: {str(e)}")

        try:
            print("Loading transformation data...")
            with open('jupyter_notebooks/outputs/engineered/full_transformation_data.pkl', 'rb') as f:
                transform_data = pickle.load(f)
                if 'feature_scaler' not in transform_data and feature_scaler:
                    transform_data['feature_scaler'] = feature_scaler
        except Exception as e:
            print(f"Warning: Could not load transformation data: {str(e)}")
            transform_data = {'encoders_and_filters': {}, 'numeric_cols': [], 'genre_columns': []}
            
        # Initialize cleaning data with defaults
        cleaning_data = {
            'frequent_crew': set(),
            'frequent_cast': set(),
            'frequent_countries': set(),
            'frequent_companies': set()
        }

        # Load top crew data with fallbacks
        top_actors = get_default_values()['top_actors']
        top_directors = get_default_values()['top_directors']
        top_writers = get_default_values()['top_writers']
        top_producers = get_default_values()['top_producers']

        try:
            with open('jupyter_notebooks/outputs/engineered/top_revenue_actors.pkl', 'rb') as f:
                top_actors = pickle.load(f)
        except Exception as e:
            print(f"Warning: Using default top actors: {str(e)}")

        try:
            with open('jupyter_notebooks/outputs/engineered/top_revenue_directors.pkl', 'rb') as f:
                top_directors = pickle.load(f)
        except Exception as e:
            print(f"Warning: Using default top directors: {str(e)}")

        try:
            with open('jupyter_notebooks/outputs/engineered/top_revenue_writers.pkl', 'rb') as f:
                top_writers = pickle.load(f)
        except Exception as e:
            print(f"Warning: Using default top writers: {str(e)}")

        try:
            with open('jupyter_notebooks/outputs/engineered/top_revenue_producers.pkl', 'rb') as f:
                top_producers = pickle.load(f)
        except Exception as e:
            print(f"Warning: Using default top producers: {str(e)}")

        class DummyPipeline:
            def transform(self, data):
                return data
                
        engineering_pipeline = DummyPipeline()

        predict_func = model.predict
        
        return (model, transform_data, cleaning_data, engineering_pipeline, predict_func, top_actors, top_directors, 
                top_writers, top_producers)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        traceback.print_exc()
        return None

def predict_movie_revenue(budget, runtime, genres, language, production_company, 
                          production_country, actor1, actor2, crew_director, 
                          crew_writer, crew_producer, popularity=0):
    try:
        data = load_data()
        if data is None:
            return None
            
        (model, transform_data, cleaning_data, engineering_pipeline, predict_func,
         _, _, _, _) = data

        raw_data = {
            'budget': budget,
            'runtime': runtime,
            'original_language': language,
            'genres': genres,
            'cast': [actor1, actor2] if actor1 and actor2 else [actor1] if actor1 else [],
            'crew': [
                f"Director_{crew_director}" if crew_director else None,
                f"Writer_{crew_writer}" if crew_writer else None,
                f"Producer_{crew_producer}" if crew_producer else None
            ],
            'production_companies': [production_company] if production_company else [],
            'production_countries': [production_country] if production_country else [],
            'popularity': popularity
        }

        input_df = pd.DataFrame([raw_data])

        input_df['crew'] = input_df['crew'].apply(lambda x: [item for item in x if item is not None])

        print("Data loaded, beginning feature engineering...")

        if 'frequent_crew' in cleaning_data:
            input_df['crew'] = input_df['crew'].apply(
                lambda x: [person for person in x if person in cleaning_data['frequent_crew']]
            )

        if 'frequent_cast' in cleaning_data:
            input_df['cast'] = input_df['cast'].apply(
                lambda x: [actor for actor in x if actor in cleaning_data['frequent_cast']]
            )

        if 'frequent_countries' in cleaning_data:
            input_df['production_countries'] = input_df['production_countries'].apply(
                lambda x: [country for country in x if country in cleaning_data['frequent_countries']]
            )

        if 'frequent_companies' in cleaning_data:
            input_df['production_companies'] = input_df['production_companies'].apply(
                lambda x: [company for company in x if company in cleaning_data['frequent_companies']]
            )

        print("Cleaning complete, starting feature engineering...")

        model_features = [f for f in transform_data['all_features'] if f != 'revenue']
        feature_df = pd.DataFrame(0, index=input_df.index, columns=model_features)

        feature_df['budget'] = input_df['budget']
        feature_df['runtime'] = input_df['runtime']
        feature_df['popularity'] = input_df['popularity']
        feature_df['budget_per_minute'] = feature_df['budget'] / feature_df['runtime']

        for genre in genres:
            genre_col = f'genre_{genre}'
            if f'genre_{genre}' in feature_df.columns:
                feature_df[f'genre_{genre}'] = 1

        if 'language_encoder' in transform_data['encoders_and_filters']:
            lang_encoder = transform_data['encoders_and_filters']['language_encoder']
            if language in lang_encoder.classes_:
                feature_df['language_encoded'] = lang_encoder.transform([language])[0]

        numeric_cols = [col for col in transform_data['numeric_cols'] if col != 'revenue']
        if 'feature_scaler' in transform_data:
            feature_df[numeric_cols] = transform_data['feature_scaler'].transform(feature_df[numeric_cols])

        feature_df = feature_df[model.feature_names_in_]
        raw_prediction = model.predict(feature_df)[0]
        print(f"Raw prediction from model: {raw_prediction}")

        predicted_revenue = raw_prediction  
        predicted_revenue = max(0, predicted_revenue) 
        
        return {
            'revenue': predicted_revenue,
            'profit': predicted_revenue - budget,
            'roi': ((predicted_revenue - budget) / budget * 100) if budget > 0 else 0,
            'is_profitable': predicted_revenue > budget,
        }
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        traceback.print_exc()
        return None

def page_predictor_body():
    st.title('Movie Revenue Predictor 🎬')
    st.info(
    f"**The client is interested in predicting the revenue and profit of a film before production**.  \n"
    f"To help make **informed investment decisions** and assess potential risks. \n"
    f"These insights will enable the client to optimize resource allocation and maximize the success of film projects."
    )

    st.subheader('Revenue predictor interface')
    
    try:
        data = load_data()
        if data is None:
            return
            
        (model, transform_data, cleaning_data, engineering_pipeline, predict_func,
         top_actors, top_directors, top_writers, top_producers) = data
        
        st.write("Enter movie details:")
        
        col1, col2 = st.columns(2) 

        with col1:
            budget = st.number_input('Budget ($)', min_value=0, value=150000000)
            runtime = st.number_input('Runtime (minutes)', min_value=0, value=120)
            
        with col2:
            language = st.selectbox('Language', transform_data['encoders_and_filters']['language_encoder'].classes_)

            production_country = st.selectbox('Production Country', 
                                           transform_data['encoders_and_filters']['frequent_countries']+ ['Other'])
            if production_country == 'Other':
                production_country = st.text_input('Enter Production Country Name')

        genres = transform_data['genre_columns']
        selected_genres = st.multiselect('Select Genres', genres)
        
        production_company = st.selectbox('Production Company', 
                                        transform_data['encoders_and_filters']['frequent_companies']+ ['Other'])
        if production_company == 'Other':
            production_company = st.text_input('Enter Production Company Name')
        
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
                    st.success('Prediction successful! 🎯')
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(label="Predicted Revenue", value=f"${int(prediction['revenue']):,}")
                    with col2:
                        st.metric(label="Profit/Loss",
                            value=f"${int(prediction['profit']):,}",
                            delta=f"${int(prediction['profit']):,}") 
                    with col3:
                        st.metric(label="ROI",
                            value=f"{prediction['roi']:.0f}%",
                            delta=f"{prediction['roi']:.0f}%")
        
                        
                    if prediction['is_profitable']:
                        st.success("🎉 This movie is predicted to be profitable!")
                    else:
                        st.warning("⚠️ This movie is predicted to lose money.")
                        
                    with st.expander("See detailed analysis", expanded=True):
                        st.subheader('Movie Details')
                        details = {
                            'Budget': f"${int(budget):,}",
                            'Expected Revenue': f"${int(prediction['revenue']):,}",
                            'Profit/Loss': f"${int(prediction['profit']):,}",
                            'Return on Investment': f"{prediction['roi']:.0f}%",
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


if __name__ == "__main__":
    page_predictor_body()
