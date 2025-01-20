# app_pages/pipeline_overview.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import pickle
import os
import joblib  

import streamlit as st
import pickle
import os

# Print current working directory to help with debugging
st.write("Current working directory:", os.getcwd())

# Define the correct base path
BASE_PATH = '/workspace/Film_Hit_prediction/jupyter_notebooks/outputs'


def load_pickle(file_path):
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path}: {str(e)}")
        return None

def display_role_metrics(role_data, role_name):
    if role_data is None:
        st.error(f"No data available for {role_name}")
        return
        
    st.subheader(f"Top {role_name} Analysis")
    
    # Show the columns (top performers)
    st.write(f"Number of top {role_name.lower()}: {len(role_data['columns'])}")
    
    # Clean the names
    clean_names = [name.split('_')[-1] for name in role_data['columns']]
    st.write(f"Top {role_name}:", ', '.join(clean_names))

    # Metrics for specific person
    selected_person = st.selectbox(
        f"Select a {role_name.lower()} to see their metrics:",
        options=list(role_data['metrics'].keys())
    )

    if selected_person:
        metrics = role_data['metrics'][selected_person]
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Movies Count", metrics['movies_count'])
            st.metric("Total Revenue", f"${metrics['total_revenue']:,.2f}")
            st.metric("Average Revenue", f"${metrics['avg_revenue']:,.2f}")
            st.metric("Hit Rate", f"{metrics['hit_rate']*100:.1f}%")
        
        with col2:
            st.metric("Average Popularity", f"{metrics['avg_popularity']:.2f}")
            st.metric("Revenue Consistency", f"${metrics['revenue_consistency']:,.2f}")
            st.metric("Composite Score", f"{metrics['composite_score']:.3f}")

def page_pipeline_overview():
    st.title("Movie Success Prediction Pipeline")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Data Cleaning Pipeline", "Feature Engineering", "Role-Based Analysis"]
    )
    
    if page == "Data Cleaning Pipeline":
        st.header("Data Cleaning Pipeline")
        
        try:
            # Load cleaning pipeline
            encoders_and_filters = load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/encoders_and_filters.pkl')
            
            if encoders_and_filters:
                # Create tabs for different aspects of cleaning
                tab1, tab2 = st.tabs(["Pipeline Steps", "Transformation Details"])
                
                with tab1:
                    st.markdown("### Pipeline Components")
                    
                    # Group pipeline steps by category
                    categories = {
                        "Encoding": [step for step in encoders_and_filters.keys() if 'mlb' in step or 'encoder' in step],
                        "Filtering": [step for step in encoders_and_filters.keys() if 'min_appearances' in step or 'positions' in step],
                        "Feature Selection": [step for step in encoders_and_filters.keys() if 'frequent' in step],
                        "Other": [step for step in encoders_and_filters.keys() if not any(x in step for x in ['mlb', 'encoder', 'min_appearances', 'positions', 'frequent'])]
                    }
                    
                    for category, steps in categories.items():
                        if steps:
                            with st.expander(f"{category} Steps", expanded=True):
                                for step in steps:
                                    col1, col2 = st.columns([1, 2])
                                    with col1:
                                        st.code(step)
                                    with col2:
                                        if 'mlb' in step:
                                            st.write(f"Multi-label binarization for {step.replace('mlb_', '')}")
                                        elif 'min_appearances' in step:
                                            st.write("Filters out rare items based on frequency threshold")
                                        elif 'frequent' in step:
                                            st.write("Selects frequently occurring items")
                                        elif 'positions' in step:
                                            st.write("Filters based on crew positions")
                                        elif 'encoder' in step:
                                            st.write("Converts categorical labels to numerical values")
                                        else:
                                            st.write("Custom feature creation step")
                
                with tab2:
                    st.markdown("### Transformation Details")
                    
                    for key, transformer in encoders_and_filters.items():
                        with st.expander(f"Transformer: {key}"):
                            if hasattr(transformer, 'classes_'):
                                st.write("Number of unique classes:", len(transformer.classes_))
                                if len(transformer.classes_) < 10:
                                    st.write("Classes:", transformer.classes_)
                            elif isinstance(transformer, (int, float)):
                                st.write("Threshold value:", transformer)
                            elif isinstance(transformer, list):
                                st.write("Number of items:", len(transformer))
                                if len(transformer) < 10:
                                    st.write("Items:", transformer)
                            else:
                                st.write("Type:", type(transformer).__name__)
        
        except Exception as e:
            st.error(f"Error in cleaning pipeline: {str(e)}")
    
    elif page == "Feature Engineering":
        st.header("Feature Engineering Pipeline")
        
        try:
            # Load feature engineering pipeline
            feature_pipeline = load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/models/movie_feature_engineering_pipeline.pkl')
            
            if feature_pipeline:
                st.markdown("### Feature Engineering Components")
                
                # Display feature engineering steps
                st.subheader("Scaling and Transformation")
                if 'feature_scaler' in feature_pipeline:
                    st.write("Feature Scaler:", type(feature_pipeline['feature_scaler']).__name__)
                
                # Show encoding information
                st.subheader("Feature Encoding")
                if 'encoders_and_filters' in feature_pipeline:
                    encoders = feature_pipeline['encoders_and_filters']
                    st.write(f"Minimum appearances threshold: {encoders.get('crew_min_appearances', 'N/A')}")
                    
                    # Show number of features for each role
                    roles = ['Director', 'Producer', 'Writer']
                    for role in roles:
                        role_columns = [col for col in encoders.get('crew_frequent_columns', []) 
                                      if f'crew_{role}_' in col]
                        st.write(f"Number of {role} features: {len(role_columns)}")
        
        except Exception as e:
            st.error(f"Error in feature engineering pipeline: {str(e)}")
    
    elif page == "Role-Based Analysis":
        st.header("Role-Based Analysis")
        
        try:
            # Load role-based analysis results
            roles_data = {
                "Actors": load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_actors.pkl'),
                "Directors": load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_directors.pkl'),
                "Producers": load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_producers.pkl'),
                "Writers": load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_writers.pkl'),
            }
            
            # Create tabs for different roles
            tabs = st.tabs(list(roles_data.keys()))
            
            for tab, (role_name, role_data) in zip(tabs, roles_data.items()):
                with tab:
                    display_role_metrics(role_data, role_name)
        
        except Exception as e:
            st.error(f"Error in role-based analysis: {str(e)}")

if __name__ == "__main__":
    st.set_page_config(
        page_title="Movie Success Prediction Pipeline",
        page_icon="🎬",
        layout="wide"
    )
    main()
  
'''

def page_pipeline_overview():
    try:
        # Load and inspect encoders and filters
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/encoders_and_filters.pkl', 'rb') as f:
            encoders_and_filters = pickle.load(f)
        
        st.markdown("### 1. Encoders and Filters Pipeline Components")
        st.markdown("**Pipeline Steps:**")

        pipeline_str = "Pipeline(steps=[\n"
        for key in encoders_and_filters.keys():
            pipeline_str += f"    ('{key}'),\n"
        pipeline_str += "])"
        st.code(pipeline_str)

        # Display description of what each step does
        st.markdown("**Pipeline Steps Description:**")
        for step in steps:
            if 'mlb' in step:
               st.write(f"- `{step}`: Multi-label binarization for {step.replace('mlb_', '')}")
            elif 'min_appearances' in step:
               st.write(f"- `{step}`: Frequency threshold filtering")
            elif 'frequent' in step:
               st.write(f"- `{step}`: Selection of frequent items")
            elif 'positions' in step:
               st.write(f"- `{step}`: Position filtering")
            elif 'encoder' in step:
               st.write(f"- `{step}`: Label encoding")
            else:
                st.write(f"- `{step}`: Feature creation")
    
    except Exception as e:
        st.error(f"Error loading pipelines: {str(e)}")
        st.error(f"Full error details: {str(type(e).__name__)}: {str(e)}")

    

      


        st.title("ML Pipeline Structure")

        st.markdown("**There are 2 ML Pipelines arranged in series.**")
        st.markdown("* The first is responsible for data cleaning and feature engineering.")
    try:
        with open('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/encoders_and_filters.pkl', 'rb') as f:
            encoders_and_filters = pickle.load(f)
    
            pipeline_str = "Pipeline(steps=[\n"
            for key in encoders_and_filters.keys():
            
        # Display pipeline components in multiple rows
                pipeline_str += f"    ('{key}', MultiLabelBinarizer()),\n" if 'mlb' in key \
                    else f"    ('{key}', FrequencyThresholdFilter()),\n" if 'min_appearances' in key \
                    else f"    ('{key}', ColumnSelector()),\n" if 'frequent' in key \
                    else f"    ('{key}', PositionFilter()),\n" if 'positions' in key \
                    else f"    ('{key}', LabelEncoder()),\n" if 'encoder' in key \
                    else f"    ('{key}', FeatureCreator()),\n"
            pipeline_str = pipeline_str.rstrip(',\n') + "\n])"
       
        st.code(pipeline_str)

        st.markdown("---")
        st.markdown("**Second Pipeline: Feature Engineering**")
        
            
        # Create pipeline string for second pipeline
        feature_pipeline_str = """Pipeline(steps=["
        components = {
            'budget_imputer': 'SimpleImputer(strategy="median")',
            'runtime_imputer': 'SimpleImputer(strategy="median")',
            'release_date_transformer': 'DateFeatureExtractor()',
            'budget_scaler': 'StandardScaler()',
            'runtime_scaler': 'StandardScaler()',
            'target_encoder': 'LogTransformer()'
        ])
    
        
        st.code(feature_pipeline_str)
       
        st.info(
            f"* This pipeline uses a Random Forest model optimized through GridSearchCV to predict movie revenues. \n"
            f"* The model achieved an R² score of 0.15, showing the challenge of predicting exact movie revenues. \n"
            f"* Budget is the most influential feature, accounting for 46.7% of prediction weight. \n"
            f"* Genre factors like Comedy (4.6%), Drama (4.4%), and Thriller (4.2%) also play significant roles. \n"
            f"* The model performs with an RMSE of $1.06 and MAE of $0.89 on logged, scaled revenue values."
        )
    '''
          
    



  
    
    



   
  

