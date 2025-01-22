import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
import pickle
import os
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns 
import sys

sys.path.append('/workspace/Film_Hit_prediction/jupyter_notebooks')

class MovieFeatureEngineeringPipeline:
    def __init__(self, feature_scaler=None, transform_data=None, actor_data=None, 
                 director_data=None, producer_data=None, writer_data=None):
        self.feature_scaler = feature_scaler
        self.transform_data = transform_data
        self.actor_data = actor_data
        self.director_data = director_data
        self.producer_data = producer_data
        self.writer_data = writer_data

BASE_PATH = '/workspace/Film_Hit_prediction/jupyter_notebooks/outputs'

def load_pickle(file_path):
    try:
        if 'movie_feature_engineering_pipeline.pkl' in file_path:
            import __main__
            class MovieFeatureEngineeringPipeline:
                def __init__(self, feature_scaler=None, transform_data=None, actor_data=None, 
                           director_data=None, producer_data=None, writer_data=None):
                    self.feature_scaler = feature_scaler
                    self.transform_data = transform_data
                    self.actor_data = actor_data
                    self.director_data = director_data
                    self.producer_data = producer_data
                    self.writer_data = writer_data
            setattr(__main__, 'MovieFeatureEngineeringPipeline', MovieFeatureEngineeringPipeline)

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

    if isinstance(role_data, dict) and 'columns' in role_data:
        st.write(f"Number of top {role_name.lower()}: {len(role_data['columns'])}")
    
    clean_names = [name.split('_')[-1] for name in role_data['columns']]
    st.write(f"Top {role_name}:", ', '.join(clean_names))

    if 'metrics' in role_data and role_data['metrics']:
        selected_person = st.selectbox(
            f"Select a {role_name.lower()} to see their metrics:",
            options=list(role_data['metrics'].keys())
    )

    if selected_person and selected_person in role_data['metrics']:
        metrics = role_data['metrics'][selected_person]
        col1, col2 = st.columns(2)
        
        with col1:
            if 'movies_count' in metrics:
                st.metric("Movies Count", metrics['movies_count'])
            if 'total_revenue' in metrics:
                st.metric("Total Revenue", f"${metrics['total_revenue']:,.2f}")
            if 'avg_revenue' in metrics:
                st.metric("Average Revenue", f"${metrics['avg_revenue']:,.2f}")
            if 'hit_rate' in metrics:
                st.metric("Hit Rate", f"{metrics['hit_rate']*100:.1f}%")
        
        with col2:
            if 'avg_popularity' in metrics:
                st.metric("Average Popularity", f"{metrics['avg_popularity']:.2f}")
            if 'revenue_consistency' in metrics:
                st.metric("Revenue Consistency", f"${metrics['revenue_consistency']:,.2f}")
            if 'composite_score' in metrics:
                st.metric("Composite Score", f"{metrics['composite_score']:.3f}")

def page_pipeline_overview():
    st.title("Movie Success Prediction Pipeline")

    st.info(
        f"* **The regression** model achieved a **strong R² score of 0.7839** on the test set, demonstrating its potential in capturing key revenue-driving factors, "
        f"but it does not predict revenue with the accuracy necessary to make an investment safe. \n\n"
        f"* The model’s performance, with a** Root Mean Squared Error (RMSE) of 78.86 million** and a **Mean Absolute Error (MAE) of $43.71 million**, "
        f"provides valuable insights into revenue predictions, but also highlights the inherent uncertainties in forecasting film revenue prior to greenlight.  \n\n "
        )
    
    BASE_PATH = '/workspace/Film_Hit_prediction/jupyter_notebooks/outputs'
    PATHS = {
        'feature_engineering': os.path.join(BASE_PATH, 'models/movie_feature_engineering_pipeline.pkl'),
        'encoders': os.path.join(BASE_PATH, 'cleaned/encoders_and_filters.pkl'),
        'top_actors': os.path.join(BASE_PATH, 'engineered/top_revenue_actors.pkl'),
        'top_directors': os.path.join(BASE_PATH, 'engineered/top_revenue_directors.pkl'),
        'top_producers': os.path.join(BASE_PATH, 'engineered/top_revenue_producers.pkl'),
        'top_writers': os.path.join(BASE_PATH, 'engineered/top_revenue_writers.pkl')
    }
   
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Pipeline Overview"

    st.sidebar.title("Navigation")
    current_page = st.sidebar.radio(
        "Go to",
        ["Pipeline Overview", "Data Cleaning Pipeline", "Feature Engineering", "Cast & Crew Engineering Pipeline"],
        key="navigation_radio",
        index=["Pipeline Overview", "Data Cleaning Pipeline", "Feature Engineering", "Cast & Crew Engineering Pipeline"].index(st.session_state.current_page)
    )

    st.header("Pipeline Details")
    col1, col2, col3, col4, = st.columns(4)

    with col1:
        if st.button("📊 Pipeline Overview", use_container_width=True):
            st.session_state.current_page = "Pipeline Overview"
            st.rerun()
    with col2:
        if st.button("🧹 Data Cleaning Pipeline", use_container_width=True):
            st.session_state.current_page  = "Data Cleaning Pipeline"
            st.rerun()
    with col3:
        if st.button("⚙️ Feature Engineering", use_container_width=True):
            st.session_state.current_page  = "Feature Engineering"
            st.rerun()()
    with col4:
        if st.button("👥 Cast & Crew Engineering Pipeline", use_container_width=True):
            st.session_state.current_page  = "Cast & Crew Engineering Pipeline"
            st.rerun()
        
    st.session_state.current_page = current_page
    
    if current_page == "Pipeline Overview":
        st.header("Model Performance")
        try:
            model_eval = load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/models/model_evaluation.pkl')
        
            if model_eval:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Root Mean Squared Error", f"${model_eval['metrics']['rmse']:,.2f}")
                    st.metric("Mean Absolute Error", f"${model_eval['metrics']['mae']:,.2f}")
                with col2:
                    st.metric("R² Score", f"{model_eval['metrics']['r2']:.4f}")
                    st.metric("Mean Absolute Percentage Error", f"{model_eval['metrics']['mape']:.2f}%")

                st.subheader("Model Visualizations")
            
                st.subheader("Predicted vs Actual Revenue")
                viz_data = model_eval['visualization_data']
                fig1 = plt.figure(figsize=(10, 6))
                plt.scatter(viz_data['actual_vs_predicted']['actual'], 
                        viz_data['actual_vs_predicted']['predicted'], 
                        alpha=0.5)
                plt.plot([viz_data['actual_vs_predicted']['actual'].min(), 
                        viz_data['actual_vs_predicted']['actual'].max()], 
                        [viz_data['actual_vs_predicted']['actual'].min(), 
                        viz_data['actual_vs_predicted']['actual'].max()], 
                        'r--', lw=2)
                plt.xlabel('Actual Revenue')
                plt.ylabel('Predicted Revenue')
                st.pyplot(fig1)

                st.subheader("Residual Plot")
                fig2 = plt.figure(figsize=(10, 6))
                plt.scatter(viz_data['residuals']['predicted'], 
                          viz_data['residuals']['values'], 
                          alpha=0.5)
                plt.axhline(y=0, color='r', linestyle='--')
                plt.xlabel('Predicted Revenue')
                plt.ylabel('Residuals')
                st.pyplot(fig2)

                st.subheader("Distribution of Residuals")
                fig3 = plt.figure(figsize=(10, 6))
                sns.histplot(viz_data['residuals_distribution']['residuals'], kde=True)
                plt.xlabel('Residuals')
                plt.ylabel('Count')
                st.pyplot(fig3)

        except Exception as e:
            st.error(f"Error loading model evaluation data: {str(e)}")

    if current_page == "Data Cleaning Pipeline":
        st.header("Data Cleaning Pipeline")
        
        try:
            encoders_and_filters = load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/encoders_and_filters.pkl')
            
            if encoders_and_filters:
                tab1, tab2 = st.tabs(["Pipeline Steps", "Transformation Details"])
                
                with tab1:
                    st.markdown("### Pipeline Components")
                    
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
    
    elif current_page == "Feature Engineering":
        st.header("Feature Engineering Pipeline")
        
        try:
            feature_pipeline = load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/models/movie_feature_engineering_pipeline.pkl')
            
            if feature_pipeline:
                tab1, tab2 = st.tabs(["Pipeline Components", "Detailed Metrics"])

                with tab1:
                    st.markdown("### Feature Engineering Components")
                
                    with st.expander("Scaling and Transformation", expanded=True):
                        if hasattr(feature_pipeline, 'feature_scaler'):
                            col1, col2 = st.columns([1, 2])
                            with col1:
                                st.code("Feature Scaler")
                            with col2:
                                st.write(type(feature_pipeline.feature_scaler).__name__)

                with st.expander("Transform Data", expanded=True):
                    if hasattr(feature_pipeline, 'transform_data'):
                        data = feature_pipeline.transform_data
                        if isinstance(data, dict):
                            for key, value in data.items():
                                st.write(f"{key}: {type(value).__name__}")
                                if isinstance(value, (list, dict)):
                                    st.write(f"Number of items: {len(value)}")
                
                with st.expander("Numeric Features", expanded=True):
                    if hasattr(feature_pipeline, 'transform_data'):
                        numeric_cols = feature_pipeline.transform_data.get('numeric_cols', [])
                        st.write(f"Number of numeric features: {len(numeric_cols)}")
                
                with st.expander("Genre Features", expanded=True):
                    if hasattr(feature_pipeline, 'transform_data'):
                        genres = feature_pipeline.transform_data.get('genre_columns', [])
                        st.write(f"Number of genre features: {len(genres)}")
                        if genres:
                            st.write("Available genres:", ", ".join(genres))

                with st.expander("Movie Metrics", expanded=True):
                    if hasattr(feature_pipeline, 'transform_data'):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Budget", "Available" if 'budget' in feature_pipeline.transform_data.get('numeric_cols', []) else "Not Available")
                        with col2:
                            st.metric("Runtime", "Available" if 'runtime' in feature_pipeline.transform_data.get('numeric_cols', []) else "Not Available")
                        with col3:
                            st.metric("Popularity", "Available" if 'popularity' in feature_pipeline.transform_data.get('numeric_cols', []) else "Not Available")
                        
                with st.expander("Cast & Crew Features", expanded=True):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Top Actors", len(feature_pipeline.actor_data.get('columns', [])))
                        st.metric("Top Directors", len(feature_pipeline.director_data.get('columns', [])))
                    with col2:
                        st.metric("Top Producers", len(feature_pipeline.producer_data.get('columns', [])))
                        st.metric("Top Writers", len(feature_pipeline.writer_data.get('columns', [])))

                with tab2:
                    st.markdown("### Role Based Metrics")
                    with st.expander("Actor Data", expanded=True):
                        if hasattr(feature_pipeline, 'actor_data'):
                            data = feature_pipeline.actor_data
                            if isinstance(data, dict):
                                st.write(f"Number of metrics: {len(data.get('metrics', {}))}")
                                st.write(f"Number of columns: {len(data.get('columns', []))}")

                    with st.expander("Director Data", expanded=True):
                        if hasattr(feature_pipeline, 'director_data'):
                            data = feature_pipeline.director_data
                            if isinstance(data, dict):
                                st.write(f"Number of metrics: {len(data.get('metrics', {}))}")
                                st.write(f"Number of columns: {len(data.get('columns', []))}")

                    with st.expander("Producer Data", expanded=True):
                        if hasattr(feature_pipeline, 'producer_data'):
                            data = feature_pipeline.producer_data
                            if isinstance(data, dict):
                                st.write(f"Number of metrics: {len(data.get('metrics', {}))}")
                                st.write(f"Number of columns: {len(data.get('columns', []))}")

                    with st.expander("Writer Data", expanded=True):
                        if hasattr(feature_pipeline, 'writer_data'):
                            data = feature_pipeline.writer_data
                            if isinstance(data, dict):
                                st.write(f"Number of metrics: {len(data.get('metrics', {}))}")
                                st.write(f"Number of columns: {len(data.get('columns', []))}")   
            
        except Exception as e:
                st.error(f"Error in feature engineering pipeline: {str(e)}")
    
    elif current_page == "Cast & Crew Engineering Pipeline":
        st.header("Cast & Crew Engineering Pipeline")
        
        try:
            roles_data = {
                "Actors": load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_actors.pkl'),
                "Directors": load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_directors.pkl'),
                "Producers": load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_producers.pkl'),
                "Writers": load_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/engineered/top_revenue_writers.pkl'),
            }
            
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
  

          
    



  
    
    



   
  

