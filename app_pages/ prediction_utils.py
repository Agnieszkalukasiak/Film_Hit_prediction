import joblib
import pickle
import pandas as pd
import numpy as np
import traceback

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
