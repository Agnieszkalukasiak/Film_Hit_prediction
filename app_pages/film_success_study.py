import plotly.express as px
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def page_film_success_study_body():

    # load data
    df_movies = pd.read_csv('encoded_movies.csv')

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

      if st.checkbox("Inspect the plots of the most peoduced Gengres "):
      # Display the plot of general film profitibility
        st.image(
            "outputs/figures/Percentage_Movies_making_a_profit_plot.png",
            caption="Percentage of Profitable Movies",
            use_column_width=True
        )

    # Divider
    st.write("---") 

    # Correlation Study Summary
    st.write(
        "* A correlation study was conducted to understand how budget, language, and genres are correlated to revenue."
    )

        # Revenue
    st.write("* A correlation study was conducted to understand how budget, language, and genres are correlated to revenue."
        )
    st.info(
        "Key Insights from the Correlation Study:\n"
        "- Revenue is most strongly correlated with **budget**.\n"
        "- Higher budgets generally result in higher revenues.\n"
        "- Genres most correlated with high revenue are: **Adventure, Action, Animation, Comedy, Fantasy, and Sci-Fi**.\n"
        "- Languages most connected to high revenue include: **English, Japanese, Telugu (South Indian), and Chinese**.\n"
        )

    # Analysing plots
    if st.checkbox("Show Correlation Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Revenue Correlation Heatmap**")
            st.image("outputs/figures/revenue_correlations_heatmap.png", 
            use_column_width=True
            )
            st.write("**Budgets Correlation To Revenue**")
            st.image("outputs/figures/Budget_vs_Revenue_scatter_points.png", 
            use_column_width=True
            )
            
        with col2:
            st.write("**Genre Correlation To Revenue**")
            st.image("outputs/figures/Genre_correlation_ with_ Revenue_plot.png"
            ,caption=" Genre Correlation To Revenue",
            use_column_width=True
            )
            st.write("**Language Correlation To Revenue**")
            st.image("outputs/figures/Average_Revenue_by_Language.png", 
            use_column_width=True
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

