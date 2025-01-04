import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_telco_data



import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")


def film_success_study_body():

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

        #Load Data
        df_movies = pd.read_csv('encoded_movies.csv')

        st.write(
            f"* The plot shows {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")
        
        df_movies = pd.read_csv('encoded_movies.csv')

        # Display the first 10 rows of the data Frame
        st.write(df.head(10))

    if st.checkbox("Inspect the plots for profitability "):

      # Display the plot of general film profitibility
        st.image(
    "/workspace/Film_Hit_prediction/outputs/figures/Percentage_of_Profitable_Movies_plot.png",
        caption="Percentage of Profitable Movies",
        use_column_width=True
    )

    if st.checkbox("Inspect the most produced genres.  "):

    # Display the plot of film production by genre 
        st.image(
    "/workspace/Film_Hit_prediction/outputs/figures/Number_of_movies_produced_by_genre.png",
        caption="Percentage of Profitable Movies",
        use_column_width=True
    )

    # Divider
    st.write("---") 

        # Revenue
        st.write(
        f"* The client is intressted in understanding the overall picture of the film industry. \n "
        f"* It's profitibility and what gengre is the most produced. \n "
    )

    # Display the plot of general film profitibility
    st.image(
     "/workspace/Film_Hit_prediction/outputs/figures/Percentage_of_Profitable_Movies_plot.png",
        caption="Percentage of Profitable Movies",
        use_column_width=True
    )

    # Divider
    st.write("---") 

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables, budget, gengre, language, are correlated to revenue. \n"
    )

    # Text based on "03 - Film Sucess Study" notebook - "Conclusions" section
    st.info(
        f"The correlation indications and plots below interpretation converge. "
        f"It is indicated that: \n"
        f"* A revenue is mostly correlated with the budget out of tge 3 variables: budget, language, gengres. \n"
        f"* The higher the budget, the higher the revenue. \n"
        f"* The genres most corelating to high revenue is Adventure, Action, Animation, Comedy, Fantasy and SiFi. \n"
        f"* The language most connected to high revenue are: English, Japanese, Telugu (south Indian), Chinese. \n"
    )
def show_analysis_plots():
    if st.checkbox("Show Correltion Analysis"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.image("outputs/figures/revenue_correlations_heatmap.png", caption="Revenue Correlation ")
            st.image("outputs/figures/Budget_vs_Revenue_scatter_points.png", caption="Budgets Correlation To Revenue")
            
        with col2:
            st.image("outputs/figures/Average_Revenue_by_Genre_plot.png", caption=" Genre Correlation To Revenue")
            st.image("outputs/figures/Average_Revenue_by_Language.png", caption="Language Correlation To Revenue")

