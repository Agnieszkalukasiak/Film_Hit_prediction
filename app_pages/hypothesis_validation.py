import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.success(
        f"* We suspect the variable of budget, language and gengre alone, do not allow for a acurate determination of revenue:Correct. "
        f"* The trianed model using the three variables to predict the outcome, had only a 15% accuracy./n"


        f"* The impact of gengre, language and budge on revenue explore in the Film_sucess_study will be used for further discussion and investigation. "
        f" The correlation between a higher budget with a higher revenue is to be futher eplored. "
    
    )
