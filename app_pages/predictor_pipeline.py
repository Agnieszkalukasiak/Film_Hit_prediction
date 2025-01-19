# app_pages/pipeline_overview.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

def load_pkl_file(filepath):
    try:
        print(f"Loading file from: {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        with open(filepath, 'rb') as file:
            data = pickle.load(file)
        return data
    except Exception as e:
        st.error(f"Could not load model: {str(e)}")
        return None

    return pipeline_data

def page_pipeline_overview():
    st.title("Data Science Pipeline Overview")
    version = 'v1'

    pickle_files = {
        'Data Cleaning': {
            'cleaning_pipeline': '/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/cleaning_pipeline.pkl'
        },
        'Feature Engineering': {
           'movie_feature_engineering_pipeline':'/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/models/movie_feature_engineering_pipeline.pkl'
        }
    }

    try:
        # Display the loaded data
        st.markdown("**There are 2 ML Pipelines arranged in series.**")

        # Process each pipeline stage
        for stage, files in pickle_files.items():
            st.markdown(f"\n## {stage} Pipeline")
            
            # Load and display each pickle file in the stage
            for file_name, filepath in files.items():
                try:
                    st.markdown(f"\n### {file_name.replace('_', ' ').title()}")
                    print(f"Loading file from: {os.path.abspath(filepath)}")

                    # Load the data
                    loaded_data = load_pkl_file(filepath)

                    if isinstance(encoders_and_filters, dict):
                    # Display pipeline steps in code format
                        st.markdown("**Components:**")
                        st.markdown(f"`Pipeline components: {list(loaded_data.keys())}`")
        
                        for key, value in encoders_and_filters.items():
                            st.markdown(f"* **{key}:**")
                            st.code(str(value))

                    elif isinstance(loaded_data, (pd.DataFrame, pd.Series)):
                    # Handle pandas DataFrames/Series
                        st.markdown("**Data Overview:**")
                        st.write(f"Shape: {data.shape}")
                        st.write("Sample of data:")
                        st.dataframe(data.head())

                    elif isinstance(data, (list, set)):
                    # Handle lists/sets (like top revenue items)
                        st.markdown("**Top Items:**")
                        st.write(f"Total items: {len(data)}")
                        st.write("Sample of items:")
                        st.code(str(list(data)[:10]))  # Show first 10 items

                    else:
                    # Handle other types of data
                        st.markdown("**Content:**")
                        st.code(str(data))

                except Exception as e:
                    st.error(f"Error processing {file_name}: {str(e)}")
                    continue 
    except Exception as e:
        st.error(f"Error in pipeline overview: {str(e)}")
        st.write("Please check the file paths and ensure all pickle files exist.")
   

if __name__ == "__main__":
    page_pipeline_overview()

# display pipeline training summary conclusions
    
st.info(
    f"* This pipeline uses a Random Forest model optimized through GridSearchCV to predict movie revenues. \n"
    f"* The model achieved an R² score of 0.15, showing the challenge of predicting exact movie revenues. \n"
    f"* Budget is the most influential feature, accounting for 46.7% of prediction weight. \n"
    f"* Genre factors like Comedy (4.6%), Drama (4.4%), and Thriller (4.2%) also play significant roles. \n"
    f"* The model performs with an RMSE of $1.06 and MAE of $0.89 on logged, scaled revenue values."
    )
