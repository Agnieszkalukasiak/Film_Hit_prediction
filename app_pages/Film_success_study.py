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
        f"* . \n"
    )

    # Code copied from "02 - Churned Customer Study" notebook - "EDA on selected variables" section
    df_eda = df.filter(vars_to_study + ['Churn'])

    # Individual plots per variable
    if st.checkbox("Churn Levels per Variable"):
        churn_level_per_variable(df_eda)

    # Parallel plot
    if st.checkbox("Parallel Plot"):
        st.write(
            f"* Information in yellow indicates the profile from a churned customer")
        parallel_plot_churn(df_eda)


# function created using "02 - Churned Customer Study" notebook code - "Variables Distribution by Churn" section
def churn_level_per_variable(df_eda):
    target_var = 'Churn'

    for col in df_eda.drop([target_var], axis=1).columns.to_list():
        if df_eda[col].dtype == 'object':
            plot_categorical(df_eda, col, target_var)
        else:
            plot_numerical(df_eda, col, target_var)


# code copied from "02 - Churned Customer Study" notebook - "Variables Distribution by Churn" section
def plot_categorical(df, col, target_var):
    fig, axes = plt.subplots(figsize=(12, 5))
    sns.countplot(data=df, x=col, hue=target_var,
                  order=df[col].value_counts().index)
    plt.xticks(rotation=90)
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# code copied from "02 - Churned Customer Study" notebook - "Variables Distribution by Churn" section
def plot_numerical(df, col, target_var):
    fig, axes = plt.subplots(figsize=(8, 5))
    sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step")
    plt.title(f"{col}", fontsize=20, y=1.05)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# function created using "02 - Churned Customer Study" notebook code - Parallel Plot section
def parallel_plot_churn(df_eda):

    # hard coded from "disc.binner_dict_['tenure']"" result,
    tenure_map = [-np.Inf, 6, 12, 18, 24, np.Inf]
    # found at "02 - Churned Customer Study" notebook
    # under "Parallel Plot" section
    disc = ArbitraryDiscretiser(binning_dict={'tenure': tenure_map})
    df_parallel = disc.fit_transform(df_eda)

    n_classes = len(tenure_map) - 1
    classes_ranges = disc.binner_dict_['tenure'][1:-1]
    LabelsMap = {}
    for n in range(0, n_classes):
        if n == 0:
            LabelsMap[n] = f"<{classes_ranges[0]}"
        elif n == n_classes-1:
            LabelsMap[n] = f"+{classes_ranges[-1]}"
        else:
            LabelsMap[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"

    df_parallel['tenure'] = df_parallel['tenure'].replace(LabelsMap)
    fig = px.parallel_categories(
        df_parallel, color="Churn", width=750, height=500)
    # we use st.plotly_chart() to render, in notebook is fig.show()
    st.plotly_chart(fig)
