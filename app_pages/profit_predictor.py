import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
import seaborn as sns 
import pickle
import numpy as np
import traceback
'''
def load_data():
    """Load all necessary models and data"""
    try:
        # Load the model
        model = joblib.load('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/models/film_revenue_model_Random Forest_20250119.joblib')
        
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
            
        return (model, transform_data, feature_scaler, 
                top_actors, top_directors, top_writers, top_producers)
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

'''

def load_data():
    """Load all necessary models and data"""
    try:
        print("Loading models and data...")
        
        # Load the trained model
        print("Loading model...")
        model = joblib.load('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/models/film_revenue_model_Random Forest_20250119.joblib')
        
        # Load the saved transformation data
        print("Loading transformation data...")
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/full_transformation_data.pkl', 'rb') as f:
            transform_data = pickle.load(f)

        # Load the cleaning and engineering pipelines
        print("Loading cleaning data...")
        cleaning_data = {}
        try:
            with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/cleaning_pipeline.pkl', 'rb') as f:
                cleaning_data = pickle.load(f)
                print("Keys in cleaning_data:", cleaning_data.keys())
        except Exception as e:
            print(f"Warning: Could not load cleaning data: {str(e)}")
        
        print("Loading engineering data...")
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/models/movie_feature_engineering_pipeline.pkl', 'rb') as f:
            engineering_pipeline = pickle.load(f)
            
        return (model, transform_data, feature_scaler, 
                top_actors, top_directors, top_writers, top_producers)
                
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None


def predict_movie_revenue(budget, runtime, genres, language, production_company, 
                          production_country, actor1, actor2, crew_director, 
                          crew_writer, crew_producer, popularity=0):
    try:
        # Create initial raw data
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

       # Create DataFrame
        input_df = pd.DataFrame([raw_data])

        # Filter out None values from crew list
        input_df['crew'] = input_df['crew'].apply(lambda x: [item for item in x if item is not None])

        print("Data loaded, beginning feature engineering...")

        # Apply cleaning based on the cleaning data dictionary
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

        # Initialize all possible feature columns with 0

        model_features = [f for f in transform_data['all_features'] if f != 'revenue']
        feature_df = pd.DataFrame(0, index=input_df.index, columns=model_features)

        # Fill in numeric values
        feature_df['budget'] = input_df['budget']
        feature_df['runtime'] = input_df['runtime']
        feature_df['popularity'] = input_df['popularity']
        feature_df['budget_per_minute'] = feature_df['budget'] / feature_df['runtime']


        # Handle genres
        for genre in genres:
            genre_col = f'genre_{genre}'
            if f'genre_{genre}' in feature_df.columns:
                feature_df[f'genre_{genre}'] = 1

        # Handle language
        if 'language_encoder' in transform_data['encoders_and_filters']:
            lang_encoder = transform_data['encoders_and_filters']['language_encoder']
            if language in lang_encoder.classes_:
                feature_df['language_encoded'] = lang_encoder.transform([language])[0]

        print("Basic features created, proceeding with final transformations...")

        # Scale numeric features if scaler exists
        numeric_cols = [col for col in transform_data['numeric_cols'] if col != 'revenue']
        if 'feature_scaler' in transform_data:
            feature_df[numeric_cols] = transform_data['feature_scaler'].transform(feature_df[numeric_cols])

        print(f"Final feature matrix shape: {feature_df.shape}")

        # Make prediction
        raw_prediction = model.predict(feature_df)[0]
        print(f"Raw prediction from model: {raw_prediction}")

        # Don't apply scaling factors - the prediction is already in the right units
        predicted_revenue = raw_prediction  # The model predicts in dollars
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


      
'''

def predict_movie_revenue(budget, runtime, genres, language, production_company, 
                          production_country, actor1, actor2, crew_director, 
                          crew_writer, crew_producer):
    try:
        print("Loading models and data...")
        
        # Load the trained model
        model = joblib.load('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/models/film_revenue_model_Random Forest_20250115.joblib')
        
        # Load the saved transformation data
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/full_transformation_data.pkl', 'rb') as f:
            transform_data = pickle.load(f)

        # Load the feature scaler separately
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/feature_scaler.pkl', 'rb') as f:
            feature_scaler = pickle.load(f)

        # Extract components from transform_data
        numeric_cols = [col for col in transform_data['numeric_cols'] if col != 'revenue']
        all_features = [col for col in transform_data['all_features'] if col != 'revenue']


        # Initialize features with zero values for all columns
        features = {col: 0 for col in all_features}

        # Process numeric features
        features['budget'] = budget
        features['runtime'] = runtime
        features['budget_per_minute'] = budget / runtime if runtime > 0 else 0
        features['popularity'] = 0 

        # Process genres
        for genre in transform_data['genre_columns']:
            features[genre] = 1 if genre in genres else 0

        # Encode language (assuming LabelEncoder was used)
        language_encoder = transform_data['encoders_and_filters']['language_encoder']
        features['language_encoded'] = language_encoder.transform([language])[0] if language in language_encoder.classes_ else 0

        # Load top actors, directors, writers, and producers data
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_actors.pkl', 'rb') as f:
            actor_data = pickle.load(f)
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_directors.pkl', 'rb') as f:
            director_data = pickle.load(f)
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_writers.pkl', 'rb') as f:
            writer_data = pickle.load(f)
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_producers.pkl', 'rb') as f:
            producer_data = pickle.load(f)

        top_actor_cols = actor_data['columns']
        top_director_cols = director_data['columns']
        top_writer_cols = writer_data['columns']
        top_producer_cols = producer_data['columns']

        # Initialize "other" counts
        other_actor_count = 0
        other_director_count = 0
        other_writer_count = 0
        other_producer_count = 0

        # Process actors
        for actor in [actor1, actor2]:
            actor_col = f"cast_{actor.replace(' ', '_')}"
            if actor_col in top_actor_cols:
                features[actor_col] = 1
        else:
                other_actor_count += 1

        # Process director
        director_col = f"crew_Director_{crew_director.replace(' ', '_')}"
        if director_col in top_director_cols:
            features[director_col] = 1
        else:
            other_director_count += 1

        # Process writer
        writer_col = f"crew_Writer_{crew_writer.replace(' ', '_')}"
        if writer_col in top_writer_cols:
            features[writer_col] = 1
        else:
            other_writer_count += 1

        # Process producer
        producer_col = f"crew_Producer_{crew_producer.replace(' ', '_')}"
        if producer_col in top_producer_cols:
            features[producer_col] = 1
        else:
            other_producer_count += 1

        # Add "other" counts to features
        features['other_actor_count'] = other_actor_count
        features['other_director_count'] = other_director_count
        features['other_writer_count'] = other_writer_count
        features['other_producer_count'] = other_producer_count


        # Extract top companies and countries from encoders_and_filters
        top_companies = transform_data['encoders_and_filters']['frequent_companies']
        top_countries = transform_data['encoders_and_filters']['frequent_countries']

        # Process production company (set to 1 if it's in the top list)
        if production_company in top_companies:
            company_col = f"company_{production_company}"
            if company_col in features:
                features[company_col] = 1

        # Process production country (set to 1 if it's in the top list)
        if production_country in top_countries:
            country_col = f"country_{production_country}"
            if country_col in features:
                features[country_col] = 1

        # Debug before creating pred_df
        print(f"Features created: {list(features.keys())}")
        print(f"Expected features (all_features): {all_features}")

        missing_features = set(all_features) - set(features.keys())
        extra_features = set(features.keys()) - set(all_features)

        print(f"Missing features: {missing_features}")
        print(f"Extra features: {extra_features}")

        # Debug feature set before creating pred_df
        print(f"Features created: {list(features.keys())}")
        print(f"Number of expected features: {len(all_features)}")

        # Create DataFrame from features
        pred_df = pd.DataFrame([features])

        print("\nBefore scaling:")
        numeric_data = pred_df[numeric_cols].copy()
        print("Numeric columns:", numeric_cols)
        print("Values:", numeric_data.iloc[0].to_dict())

        # Scale numeric columns
        pred_df[numeric_cols] = feature_scaler.transform(pred_df[numeric_cols])

        scaled_numeric = feature_scaler.transform(pred_df[numeric_cols])
        print("\nAfter scaling:")
        scaled_df = pd.DataFrame(scaled_numeric, columns=numeric_cols)
        print(scaled_df.iloc[0].to_dict())

        # Ensure correct column order
        pred_df = pred_df[all_features]

        # Debug feature count
        print(f"Final number of features in pred_df: {pred_df.shape[1]}")

        # Make prediction
        raw_prediction = model.predict(pred_df)[0]
        print(f"Raw prediction from model: {raw_prediction}")

       # Don't apply scaling factors - the prediction is already in the right units
        predicted_revenue = raw_prediction  # The model predicts in dollars
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

'''
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



page_predictor_body()
