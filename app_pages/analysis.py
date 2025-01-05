import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


def page_analysis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.success(
        f"* We suspect customers are churning with low tenure levels: Correct. "
        f"The correlation study at Churned Customer Study supports that. \n\n"

 



        f"* A customer survey showed our customers appreciate fibre Optic. "
        f"A churned user typically has Fibre Optic, as demonstrated by a Churned Customer Study. "
        f"This insight will be used by the survey team for further discussions and investigations."
    )

    st.write("### Movie Revenue Analysis")

    # Load visualization PNGs from your figures folder

    if st.checkbox("Feature Importance"):
        st.write("Most important features for predicting revenue")
        st.image("outputs/figures/revenue_correlations_heatmap.png")


    if st.checkbox("Budget vs Revenue Analysis"):
        st.write("Key findings about budget-revenue relationship")
        st.image("outputs/figures/Budget_vs_Revenue_scatter_points.png")
       

    if st.checkbox("Language Analysis"):
        st.write("Correlation between language and revenue")
        st.image("outputs/figures/Average_Revenue_by_Language.png")
       

    if st.checkbox("Genre Analysis"):
        st.write(" Correlation between genres and their revenue patterns")
        st.image("outputs/figures/Genre_correlation_ with_ Revenue_plot.png")
        

    
        

  