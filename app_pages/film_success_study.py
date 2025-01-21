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
 
    if st.checkbox("Inspect the correlation with revenue"):
        df = pd.read_csv("/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/correlations_with_revenue_postproduction.csv")
        st.dataframe(df, use_container_width=True) 
        
        st.image(
            "jupyter_notebooks/outputs/figures/before_greenlight_correlations_study.png",
            caption="Correlation plot with variables available before greenlight",
            use_container_width=True
        )

    if st.checkbox("Inspect Correlation Budget Revenue"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/budget_vs_revenue.png",
            use_container_width=True
        )

    if st.checkbox("Inspect High and Low Budget Movies Percentage"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/Percentage_Movies_making_a_profit_plot.png",
            use_container_width=True
        )

    if st.checkbox("Inspect Movies Making Profit Percentage "):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/Profitable_movies.png",
            use_container_width=True
        )
    
    if st.checkbox("Inspect Movies Revenue Ouliers"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/revenue_outliers.png",
            use_container_width=True
        )
    
     
    
    if st.checkbox("Inspect Correlation Genre Revenue"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/Genre_ Revenue_corr.png",
            use_container_width=True
        )
    
    if st.checkbox("Inspect Genre Produced"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/Genre_produced.png",
            use_container_width=True
        )
    
    if st.checkbox("Inspect Correlation Language Revenue"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/Language_revenue_corr.png",
            use_container_width=True
        )

    if st.checkbox("Inspect Correlation Runtime Revenue"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/runtime_revenue.png",
            use_container_width=True
        )
    
    if st.checkbox("Inspect Correlation Production Company Revenue"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/company_revenue.png",
            use_container_width=True
        )
    
    if st.checkbox("Inspect Correlation Production Country Revenue"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/country_revenue.png",
            use_container_width=True
        )

    if st.checkbox("Inspect Correlation Between Above The Line Positions and Revenue"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/most_important_abovetheline_creatives_revenue.png",
            use_container_width=True
        )
      

    if st.checkbox("Inspect Correlation Cast Popularity"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/cast_popularity.png",
            use_container_width=True
        )
    if st.checkbox("Inspect Correlation Cast and Revenue"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/cast_revenue.png",
            use_container_width=True
        )

    if st.checkbox("Inspect Correlation Director Revenue"):
        st.image(
            "/workspace/Film_Hit_prediction/jupyter_notebooks/outputs/figures/directors_revenue.png",
            use_container_width=True
        )
    if st.checkbox("Inspect Correlation Producer Revenue"):
        st.image(
            "jupyter_notebooks/outputs/figures/producer_revenue.png",
            use_container_width=True
        )
    if st.checkbox("Inspect Correlation Writer Revenue"):
        st.image(
            "jupyter_notebooks/outputs/figures/writer_revenue.png",
            use_container_width=True
        )
