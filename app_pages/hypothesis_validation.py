import streamlit as st


def page_hypothesis_validation_body():

    st.write("# Project Hypothesis and Validation")
    st.write("## Initial Hypothesis ")

    # Hypothesis
    st.success(
        "Movie revenue could be predicted primarily using budget, language, and genre.")

    st.write("### Results Disproved This Hypothesis:")

    st.info (
        f"The low R² score (0.15) indicates these variables alone cannot reliably predict revenue.\n"
        f"Other crucial factors are likely missing, such as:\n"
        f"* Star power/cast. \n"
        f"* Release timing/season. \n"
        f"* Marketing budget.\n"
        f"* Critical reviews.\n"
        f"* Competition at release.\n"
        f"* Director/studio reputation.\n"
        )
    
    st.write("### Validation:")

    st.info(
        "Model Performance Shows limited predictive power:\n"
        "* R² score of 0.15 indicates poor predictive ability.\n"
        "* Model misses 85% of what influences revenue.\n"
        "* Cannot make reliable revenue predictions with current features.\n"
    )

     
    st.write("### Learning Outcomes:")
    st.info(
        "* Budget, language, and genre alone are insufficient predictors.\n"
        "* Need more diverse and influential features.\n"
        "* Movie success is more complex than initially assumed.\n"
    )

    st.write("### Business Implications:")
    st.info(
        "* Current model should not be used for serious financial decisions.\n"
        "* More data collection needed for meaningful predictions.\n"
        "* Demonstrates complexity of movie revenue forecasting\n" 
    )

        
    
    
