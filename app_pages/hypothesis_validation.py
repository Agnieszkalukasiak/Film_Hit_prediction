import streamlit as st


def page_hypothesis_validation_body():

    st.title("Project Hypothesis and Validation")

    st.subheader("Initial Hypothesis ")

    st.write("### Correct ✅\n")
    st.info(
        "We suspected that the variables available prior to greenlighting were not sufficient to determine a film's revenue."
        "**Correct**.The correlation study between revenue and the variables available **before greenlighting** confirms that while certain factors—such as **budget, cast popularity, genre, runtime, and crew**—have a measurable impact, they do not fully determine a film's financial success.\n"
        "External factors such as **market trends, audience reception, competitive releases, and marketing budgets** contribute significantly to a film's revenue predictability."
        )

    # Actionable insight section
    st.subheader("Actionable Insight 🔍")

    st.success(
        "This finding highlights the importance of supplementing **quantitative analysis** with **qualitative assessments, industry expertise, and market research** to enhance investment decisions.\n\n"
        "Further refinement of predictive models may include additional factors such as:\n"
        "- Marketing budget\n"
        "- Distribution strategy\n"
        "- Audience sentiment\n"
        "- Social media buzz\n\n"
        "These improvements can help to enhance revenue forecasting accuracy."
    )

   

   
    

        
    
    
