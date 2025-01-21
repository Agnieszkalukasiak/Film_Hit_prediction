import plotly.express as px
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def page_film_success_study_body():

    # load data
    df_movies = pd.read_csv('workspace/Film_Hit_prediction/jupyter_notebooks/outputs/datasets/collection/merged_movie_data.csv')

    # hard copied from film sucess study customer study notebook
    vars_to_study = ['Budget', 'language',
                     'genres']

    st.write("### Film Sucess Study")
    st.info(
        f"* The client is interested in understanding the overall picture of the film industry."
        f" It's profitibility and what gengre is the most produced.. "
        )

    # inspect data
    if st.checkbox("Inspect the data the study is based on "):
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
    st.info(
        "Key Insights:\n"
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