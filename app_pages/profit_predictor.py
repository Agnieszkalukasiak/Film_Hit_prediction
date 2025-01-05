import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
import seaborn as sns 

def page_predictor_body():
    st.title("Movie Revenue Prediction")

    # load needed files
    try:
        engineered_path = "outputs/datasets/engineered/"
        model_path = "outputs/models/"
    
    #Load components
        le_language = joblib.load(os.path.join(engineered_path, 'language_encoder.joblib'))
        scaler = joblib.load(os.path.join(engineered_path, 'budget_scaler.joblib'))
        scaler_y = joblib.load(os.path.join(engineered_path, 'revenue_scaler.joblib'))
        model = joblib.load(os.path.join(model_path, 'movie_revenue_predictor.joblib'))

    # Introduction
        st.info(
            f"* The client is interested in predicting the revenue of a given film based on its budget, language, and genre. \n"
            f"* Additionally the clien is intressted if the film will break even. \n"
        )

    # Input Section
        st.header("Enter Movie Details")

    # Sample input
        budget = st.number_input("Movie Budget ($)", 
                    min_value=100000, 
                    max_value=500000000, 
                    value=1000000,
                    step=100000,
                    format="%d")
   
        language = st.selectbox("Movie Language", 
                        options=le_language.classes_,
                        index=list(le_language.classes_).index('en'))
        
        all_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
            'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
            'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
            'TV Movie', 'Thriller', 'War', 'Western', 'Foreign']
            
        genres = st.multiselect("Select Genres", 
            options=all_genres,
            default=['Action'])

        # Make prediction when button is clicked
        if st.button("Predict Revenue"):
            # Feature Processing
            st.header("Feature Processing")
            
        # Budget processing
            budget_logged = np.log1p(budget)
            budget_scaled = scaler.transform([[budget_logged]])[0][0]

                
        
        # Show processing steps in an expander
            with st.status("See processing details"):
                st.write("Budget Processing:")
                st.write(f"Original Budget:${budget:,.2f}")             
                st.write(f"Log Transformed: {budget_logged:.2f}")
                st.write("Scaled Budget:{budget_scaled:.2f}")

        # Language processing
                st.write("Language Encoding:")
                language_encoded = le_language.transform([language])[0]
                st.write(f"'{language}' encoded as: {language_encoded}")
            
        # Genre processing
            genre_dict = {genre: 1 if genre in genres else 0 for genre in all_genres}
            
        # Create feature array
            features = {
                'language_encoded': language_encoded,
                'budget_scaled': budget_scaled,
                **genre_dict
            }
            input_df = pd.DataFrame([features])

        # Make Prediction
            st.header("Prediction Results")
            
            predicted_revenue = model.predict(input_df)[0]
            predicted_revenue = scaler_y.inverse_transform([[predicted_revenue]])[0][0]
            final_revenue = budget * (1.5 + predicted_revenue)
            
            profit_loss = final_revenue - budget
            roi = (profit_loss / budget) * 100 if budget > 0 else 0
            
        # Display results
            col1, col2, col3 = st.columns(3)
            col1.metric("Predicted Revenue", f"${final_revenue:,.2f}")
            col2.metric("Profit/Loss", f"${profit_loss:,.2f}")
            col3.metric("ROI", f"{roi:.1f}%")

            
        # Feature Importance for current prediction
            st.subheader("Feature Importance")
            fig, ax = plt.subplots(figsize=(10, 6))
            feature_importance = pd.DataFrame({
                'Feature': input_df.columns,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            sns.barplot(data=feature_importance.head(10), x='Importance', y='Feature')
            plt.title("Top 10 Most Important Features")
            st.pyplot(fig)
            plt.close()
     
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug info:")
        st.write(f"Current working directory: {os.getcwd()}")

if __name__ == "__main__":
    page_predictor_body()
       


