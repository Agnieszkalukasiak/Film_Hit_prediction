import streamlit as st
import pickle
import os

# Print current working directory to help with debugging
st.write("Current working directory:", os.getcwd())

# Define the correct base path
BASE_PATH = '/workspace/Film_Hit_prediction/jupyter_notebooks/outputs'

# Define file paths
PATHS = {
    'feature_engineering': os.path.join(BASE_PATH, 'models/movie_feature_engineering_pipeline.pkl'),
    'encoders': os.path.join(BASE_PATH, 'cleaned/encoders_and_filters.pkl'),
    'top_actors': os.path.join(BASE_PATH, 'feature_engineering/top_revenue_actors.pkl'),
    'top_directors': os.path.join(BASE_PATH, 'feature_engineering/top_revenue_directors.pkl'),
    'top_producers': os.path.join(BASE_PATH, 'feature_engineering/top_revenue_producers.pkl'),
    'top_writers': os.path.join(BASE_PATH, 'feature_engineering/top_revenue_writers.pkl')
}

# Verify paths exist
st.write("Checking file paths:")
for name, path in PATHS.items():
    st.write(f"{name}: {'✅ Exists' if os.path.exists(path) else '❌ Not found'} at {path}")
    