import streamlit as st
from app_pages.multipage import MultiPage

# Load page scripts
from app_pages.page_analysis import page_analysis_body
from app_pages.page_hypothesis_validation import page_hypothesis_validation_body
from app_pages.page_predictor_pipeline import page_predictor_pipeline_body
from app_pages.page_profit_predictor import page_profit_predictor_body
from app_pages.page_project_summary import page_project_summary_body
from app_pages.page_study import page_study_body

# Create an instance of the MultiPage app
app = MultiPage(app_name="My Application")  

# Add your app pages here
app.add_page("Quick Project Summary", page_project_summary_body)
app.add_page("Analysis", page_analysis_body)
app.add_page("Hypothesis Validation", page_hypothesis_validation_body)
app.add_page("Predictor Pipeline", page_predictor_pipeline_body)
app.add_page("Profit Predictor", page_profit_predictor_body)
app.add_page("Study", page_study_body)

# Run the app
app.run()