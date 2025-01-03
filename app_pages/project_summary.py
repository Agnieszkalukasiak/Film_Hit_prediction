import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* **Revenue** : The total money a film has earned. . \n"
        f"* **Budget**: The amount of money required to produce a film. \n"
        f"* **Genres**: Different categories or types of films.\n "
        f"* **Language**: The language in which the film is made and the actors speak.\n "
        f"* ** Profit**: The money a film earns after covering production costs.\n"
    
        f"**Project Dataset**\n"
        f"* The dataset represents **Kaggle's TMDB movie data**, "
        f"containing information for 5,000 movies, including their revenues, budgets, languages, genres, production companies,"
        f"IDs, titles, cast, and crew.\n")
     

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Code-Institute-Solutions/churnometer).")
    

    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1. The client is interested in understanding the corelation between revenue and budget, language and gengre,"
        f"so that they can learn how these variables correlate with revenue.\n"
        f"* 2. The client is interested in predicting the revenue and profit of a film based on genre, language, and budget."
        )

        