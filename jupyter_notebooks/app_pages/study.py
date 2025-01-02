import streamlit as st

def page_summary_body():

    st.write("### Quick Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Terms & Jargon**\n"
        f"* A **revenue** is the money a film has made in total . \n"
        f"* A **budget** i the money it takes to produce a film. \n"
        f"**Gengres** are different types of films. \n "
        f"**Language** is the language in which the film is made and the actors speak. \n "
        f"** Profit**  is the money the film makes on top of the money spent.  " 
    
        f"**Project Dataset**\n"
        f"* The dataset represents a **customer base from a Telco company** "
        f"containing individual customer data on the products and services "
        f"(like internet type, online security, online backup, tech support), "
        f"account information (like contract type, payment method, monthly charges) "
        f"and profile (like gender, partner, dependents).")

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Code-Institute-Solutions/churnometer).")
    

    # copied from README file - "Business Requirements" section
    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in understanding the patterns from the customer base "
        f"so that the client can learn the most relevant variables that are correlated to a "
        f"churned customer.\n"
        f"* 2 - The client is interested in determining whether or not a given prospect will churn. "
        f"If so, the client is interested to know when. In addition, the client is "
        f"interested in learning from which cluster this prospect will belong in the customer base. "
        f"Based on that, present potential factors that could maintain and/or bring  "
        f"the prospect to a non-churnable cluster."
        )

        