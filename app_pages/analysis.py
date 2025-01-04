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


def page_analysis_body():
    st.write("### Movie Revenue Analysis")

    # Load visualization PNGs from your figures folder
    with st.expander("Budget vs Revenue Analysis"):
        st.image("figures/budget_revenue_correlation.png")
        st.write("Key findings about budget-revenue relationship")

    with st.expander("Language Analysis"):
        st.image("figures/language_distribution.png")
        st.write("Distribution of movies by language and their performance")

    with st.expander("Genre Analysis"):
        st.image("figures/genre_performance.png")
        st.write("Analysis of different genres and their revenue patterns")

    with st.expander("Feature Importance"):
        st.image("figures/feature_importance.png")
        st.write("Most important features for predicting revenue")

  