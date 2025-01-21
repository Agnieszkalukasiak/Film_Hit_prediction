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
    vars_to_study = ['Budget','genre', 'cast', 'runtime',
    'producer','director', 'production company', 'production country', 'popularity of elemnts atatched',]

    st.write("### Film Sucess Study")
    st.info(
    f"* The client wants to identify the key factors most strongly linked to a film's revenue potential \n"
    f"  using only the data available before the film is greenlit, to better assess its investment appeal \n"
    f"  and uncover valuable investment cues from the correlation study. \n"
    )

    # inspect data
    if st.checkbox("Inspect the initial film data "):
        st.write(f"The dataset contains {df_movies.shape[0]} rows and {df_movies.shape[1]} columns.")
        st.write("Here are the first 10 rows of the dataset:")
        st.write(df_movies.head(10))

    '''
    if st.checkbox("Inspect the plots for profitability "):
      # Display the plot of general film profitibility
        st.image(
            "outputs/figures/Percentage_Movies_making_a_profit_plot.png",
            caption="Percentage of Profitable Movies",
            use_column_width=True
        )
    '''

    # Correlation Study Summary
    st.write("### Revenue Correlation Analysis")
    st.write(
    f"* A correlation study was conducted in the notebook to better understand how "
    f"the available variables are correlated to a film's revenue potential. \n"
    f"The most relevant variables identified are: **{vars_to_study}**"
)

   
    st.info(
    f"The correlation indications and plots below interpretation converge. It is indicated that:\n\n"
    
    f"Many of the key variables influencing film revenue are the ones known after the film is produced, such as **vote count popularity**, **vote average**,"
    f"and **audience engagement**. As these variables are not known until the film is done,"
    f" it is hard to predict revenue with certainty, making films a **high risk investment** .\n\n"

    f"**Budget-Driven Success**  \n"
    f"* Only 25% of movies have a **high budget**, yet films with larger budgets tend to generate **higher box office**, "
    "indicating that producers need to carefully balance investment and expected return. However, this also makes the industry more prone to financial **risk**.  \n\n"
    f"* Around 29% of movies fall into the **low budget** category, highlighting that films with limited budgets often generate **low revenue**. "
    f"Despite the increased risks associated with low-budget films, there may still be **outliers that perform exceptionally well**, driven by factors such as "
    f"creative talent, marketing strategies, or audience appeal.  \n\n"

    f"**Profitability in the Film Industry** \n\n"
    f"* Only **53%** of movies make a profit, which highlights the inherent risks of investing in film production. "
    f"This aligns with the challenges faced by producers in trying to predict a film’s commercial success accurately.  \n\n"

    f"**Impact of Genre on Revenue** \n\n" 
    f"* Certain genres perform significantly better than others in terms of revenue. **Adventure** films are the most profitable, followed by **Fantasy**. "
    f"Genres like **Drama** tend to underperform at the box office, showing that genre choice plays a key role in the film's commercial success. \n\n "
    f"* There appears to be a correlation between the **most produced genres** and **lower box office performance**. For example, Drama, which is the most frequently "
    f"produced genre, also performs the worst, suggesting an **oversaturated market**. \n"
    f"* Investing in less frequently produced genres, such as Fantasy, would be recomended. \n\n"

    f"**Language and Revenue**\n\n"
    f"* Language appears to have less impact on revenue overall, but films in **English**, **Japanese**, **Indian**, and **Chinese** languages "
    f"tend to perform significantly better than films in other languages. This highlights the global appeal of these languages in the film industry.\n\n"
    
    f"**Runtime and Revenue**\n\n"
    f"* **Runtime** seems to have a significant impact on revenue. Films with runtimes between **100 and 130 minutes** tend to perform best at the box office. "
    f"This suggests an optimal film length for audience engagement, balancing pacing and content depth.\n\n"
    
    f"**Film Companies' Role in Revenue**\n\n"
    f"* Major film studios tend to have a high correlation with revenue. However, surprisingly, some **independent and artistic companies** such as "
    f"**Eon Productions** and **Relativity** also show strong links to film success, demonstrating that financial backing and creative risk-taking "
    f"can lead to profitability.\n\n"
    
    f"**Impact of Production Country on Revenue**\n\n"
    f"* Whether a film is produced in the US or not seems to have a significant impact on revenue, "
    f"clearly reflecting the dominance of American films in the global film industry.\n\n"

    f"**The Importance of Cast in Film Revenue**\n\n"
    f"* Among the **above the line** functions, **cast** plays the most crucial role in determining a film's **revenue**. A strong cast can drive audience engagement, "
    f"especially when the actors are associated with successful franchises or high-budget films.\n\n"
    
    f"* The relationship between cast and revenue is more complex than just hiring the most popular actors. While actors like **Jack Nicholson**, **Robert De Niro**, "
    f"and **Bruce Willis** are highly popular, actors like **Stan Lee**, **Hugo Weaving**, and **John Ratzenberger** show a stronger correlation with revenue, "
    f"likely due to their association with high-budget productions. \n\n"
    
    f"**Director Impact**\n\n"
    f"* The **above the line crew**, particularly **directors**, significantly impact revenue. Renowned directors such as **James Cameron**, **Peter Jackson**, "
    f"**Michael Bay**, and **Steven Spielberg** consistently top the list of most successful films, suggesting that their involvement "
    f"increases a film's likelihood of **high revenue**.\n\n"
    
    f"**Producers and Revenue**\n\n"
    f"* **Producers** also have a **significant correlation with revenue**, with notable names such as **Kevin Feige, James Cameron, and Peter Jackson** "
    f"often serving as the primary driving force behind a film's success. This highlights the critical importance of a producer’s track record "
    f"in predicting a film’s financial performance.\n\n"
    
    f"**Writers' Influence on Revenue**\n\n"
    f"* Writers have a more **complex relationship with revenue**. Highly successful writers such as **M. Night Shyamalan and Quentin Tarantino** show a strong "
    f"correlation with high revenue, whereas even well-known writers like **Woody Allen** exhibit a negative relationship with revenue despite their popularity. "
    f"Since the film itself is a reflection of the writer's work, this suggests that the connection between a writer and revenue is more intricate "
    f"and requires a deeper, separate analysis."

)

st.markdown("### 7. **Language and Revenue**")
st.write(
    "Language appears to have less impact on revenue overall, but films in **English**, **Japanese**, **Indian**, and **Chinese** languages "
    "tend to perform significantly better than films in other languages. This highlights the global appeal of these languages in the film industry."
)

st.markdown("### 8. **Runtime and Revenue**")
st.write(
    "**Runtime** seems to have a significant impact on revenue. Films with runtimes between **100 and 130 minutes** tend to perform best at the box office. "
    "This suggests an optimal film length for audience engagement, balancing pacing and content depth."
)

st.markdown("### 9. **Film Companies' Role in Revenue**")
st.write(
    "Major film studios tend to have a high correlation with revenue. However, surprisingly, some **independent and artistic companies** such as "
    "**Eon Productions** and **Relativity** also show strong links to film success, demonstrating that financial backing and creative risk-taking "
    "can lead to profitability."
)

st.markdown("### 10. **Impact of Country on Revenue**")
st.write(
    "The country most strongly linked to high revenue production is the **United States**, clearly reflecting its dominance in the global film industry."
)

st.markdown("### 11. **The Importance of Cast in Film Revenue**")
st.write(
    "Among the 'above the line' functions, **cast** plays the most crucial role in determining a film's revenue. A strong cast can drive audience engagement, "
    "especially when the actors are associated with successful franchises or high-budget films."
)

st.markdown("### 12. **Actor Influence on Revenue**")
st.write(
    "The relationship between cast and revenue is more complex than just hiring the most popular actors. While actors like **Jack Nicholson**, **Robert De Niro**, "
    "and **Bruce Willis** are highly popular, actors like **Stan Lee**, **Hugo Weaving**, and **John Ratzenberger** show a stronger correlation with revenue, "
    "likely due to their association with high-budget productions."
)
st.write("**Yet, popularity of cast and crew is one of the key factors driving revenue.**")

st.markdown("### 13. **Director Impact**")
st.write(
    "The 'above the line' crew, particularly directors, significantly impact revenue. Renowned directors such as **James Cameron**, **Peter Jackson**, "
    "**Michael Bay**, and **Steven Spielberg** consistently top the list of most successful directors, suggesting that their involvement "
    "increases a film's likelihood of high revenue."
)

st.markdown("### 14. **Producers and Revenue**")
st.write(
    "Producers also have a **significant correlation with revenue**, with notable names such as **Kevin Feige, James Cameron, and Peter Jackson** "
    "often serving as the primary driving force behind a film's success. This highlights the critical importance of a producer’s track record "
    "in predicting a film’s financial performance."
)

st.markdown("### 15. **Writers' Influence on Revenue**")
st.write(
    "Writers have a more **complex relationship with revenue**. Highly successful writers such as **M. Night Shyamalan and Quentin Tarantino** show a strong "
    "correlation with high revenue, whereas even well-known writers like **Woody Allen** exhibit a negative relationship with revenue despite their popularity. "
    "Since the film itself is a reflection of the writer's work, this suggests that the connection between a writer and revenue is more intricate "
    "and requires a deeper, separate analysis."
)

'''
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
    '''