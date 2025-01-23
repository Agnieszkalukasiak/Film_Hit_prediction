import streamlit as st

def page_project_summary_body():
    st.title("Project Summary")

    st.success(
        f"**Project Dataset** \n\n"
        f"The dataset is sourced from **Kaggle's TMDB movie data**, "
        f"which consists of two combined datasets containing information on **5,000 movies**. \n\n "
        f"The dataset includes details such as budget, genres, homepage, id, keywords, original_language, "
        f"original_title, overview, popularity, production_companies, production_countries, release_date, "
        f"revenue, runtime, spoken_languages, status, tagline, title_x, vote_average, vote_count, title_y, cast, and crew. \n\n"
        )
        
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* **Revenue** : The total money a film has earned. \n"
        f"* **Budget**: The amount of money required to produce a film. \n"
        f"* **Genres**: Different categories or types of films. \n"
        f"* **Language**: The language in which the film is made and the actors speak. \n"
        f"* **Profit**: The money a film earns after covering production costs. \n"
        f"* **Popularity**: A metric representing how well-known or widely recognized a film is among audiences.\n"
        f"* **Runtime**: The total duration of the film, usually measured in minutes.\n"
        f"* **Production Country**: The country where the film is produced or primarily filmed.\n"
        f"* **Production Company**: The company or studio responsible for producing and distributing the film.\n"
        f"* **Director**: The person who directs the artistic and creative aspects of the film, guiding the cast and crew.\n"
        f"* **Cast**: The group of actors who perform in the film, including lead and supporting roles.\n"
        f"* **Writer**: The person or group of people who write the script or screenplay for the film.\n"
        f"* **Producer**: The individual or company responsible for overseeing the production of the film, from conception to completion.\n"
        f"* **Greenlight**: The moment a film is cleared for starting the production process, signifying that all necessary approvals, funding, and logistics are in place.\n"
        f"* **Above the line positions**: These are the key creative roles (like **Writer**, **Producer**, **Director**) that are determined before production begins." 
        f"They are responsible for the overall creative vision and strategic direction of the film.\n"
    )
        
     
    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README filfit e](https://github.com/Agnieszkalukasiak/Film_Hit_prediction/blob/main/README.md).")
    

    # copied from README file - "Business Requirements" section
    st.success(
        f"**The project has 2 business requirements**\n"
        f"* 1. The client is interested in predicting the **revenue** and **profit** of a film before the **greenlight**.\n "
        f"* 2. The client is interested in understanding the correlation between the **variables** available before a film is greenlit and its **revenue**, "
        f"in order to assess whether a film is a good **investment opportunity** and which **variables** might be indicators.\n"
    )
