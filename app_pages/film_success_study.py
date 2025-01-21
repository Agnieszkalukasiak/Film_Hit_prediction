import plotly.express as px
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def page_film_success_study_body():

    # load data
    df_movies = pd.read_pickle('/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/cleaned/df_final_cleaned.pkl')

    # hard copied from film sucess study customer study notebook
    vars_to_study = ['Budget','runtime', 'genre', 'cast', 'crew',]

    st.write("### Film Sucess Study")
    st.info(
        f"* The client wants to analyze past film performances to identify the key factors most strongly linked to a film's revenue potential, \n" 
        f"using only the variables available before a film is greenlit for production.\n "
        )

    # inspect data
    if st.checkbox("Inspect the initial data "):
        st.write(f"The dataset contains {df_movies.shape[0]} rows and {df_movies.shape[1]} columns.")
        st.write("Here are the first 10 rows of the dataset:")
        st.write(df_movies.head(10))

    if st.checkbox("Inspect the plots for profitability "):
      # Display the plot of general film profitibility
        st.image(
            "outputs/figures/Percentage_Movies_making_a_profit_plot.png",
            caption="Percentage of Profitable Movies",
            use_column_width=True
        )

        if st.checkbox("Inspect the plots of the most produced genres "):
        # Display the plot of general film profitibility
            st.image(
                "outputs/figures/Percentage_Movies_making_a_profit_plot.png",
                caption="Percentage of Profitable Movies",
                use_column_width=True
            )

            if st.checkbox("View Genre Distribution"):
                st.image(
                    "outputs/figures/genre_distribution_plot.png",
                    caption="Most Produced Genres",
                    use_column_width=True
                )

    # Correlation Study Summary
    # Correlation Study
    st.write("### Revenue Correlation Analysis")
    st.write(
    f"* A correlation study was conducted in the notebook to better understand how "
    f"the available variables are correlated to a film's revenue potential. \n"
    f"The most relevant variables identified are: **{vars_to_study}**"
)
    st.info(
        "The correlation indications and plots below interpretation converge. It is indicated that:\n"
        "- Revenue correlates strongest with **budget**\n"
        "- High-revenue genres: **Adventure, Action, Animation, Comedy, Fantasy, Sci-Fi**\n"
        "- High-revenue languages: **English, Japanese, Telugu, Chinese**"
    )
    

     # Analysis sections
    revenue_sections = {
        "Feature Importance": {
            "title": "Most important features for predicting revenue",
            "image": "outputs/figures/revenue_correlations_heatmap.png"
        },
        "Budget vs Revenue": {
            "title": "Budget-revenue relationship analysis",
            "image": "outputs/figures/Budget_vs_Revenue_scatter_points.png"
        },
        "Language Analysis": {
            "title": "Language correlation with revenue",
            "image": "outputs/figures/Average_Revenue_by_Language.png"
        },
        "Genre Analysis": {
            "title": "Genre revenue patterns",
            "image": "outputs/figures/Genre_correlation_with_Revenue_plot.png"
        }
    }

    for section, data in revenue_sections.items():
        if st.checkbox(f"Show {section}"):
            st.write(data["title"])
            st.image(data["image"], use_column_width=True)