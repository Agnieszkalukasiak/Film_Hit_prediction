import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import joblib
import numpy as np
import seaborn as sns 

def page_predictor_pipeline_body():
    st.title("Movie Revenue Prediction Pipeline")

    # load needed files
    try:
        engineered_path = "outputs/datasets/engineered/"
        model_path = "outputs/models/"
    
    #Load components
        le_language = joblib.load(os.path.join(engineered_path, 'language_encoder.joblib'))
        scaler = joblib.load(os.path.join(engineered_path, 'budget_scaler.joblib'))
        scaler_y = joblib.load(os.path.join(engineered_path, 'revenue_scaler.joblib'))
        model = joblib.load(os.path.join(model_path, 'movie_revenue_predictor.joblib'))


    # display pipeline training summary conclusions
        st.info(
            f"* The pipeline was tuned aiming at least 0.80 Recall on 'Yes Churn' class, "
            f"since we are interested in this project in detecting a potential churner. \n"
            f"* The pipeline performance on train and test set is 0.90 and 0.85, respectively."
        )

        st.write("### ML Pipeline Steps")

    # Sample input
        st.header("1. Enter Movie Details")

        col1, col2 = st.columns(2)
        with col1:
            budget = st.number_input("Movie Budget ($)", 
                                   min_value=100000, 
                                   max_value=500000000, 
                                   value=1000000,
                                   step=100000,
                                   format="%d")
   
            language = st.selectbox("Movie Language", 
                                  options=le_language.classes_,
                                  index=list(le_language.classes_).index('en'))
        
        with col2:
            all_genres = ['Action', 'Adventure', 'Animation', 'Comedy', 'Crime',
                         'Documentary', 'Drama', 'Family', 'Fantasy', 'History',
                         'Horror', 'Music', 'Mystery', 'Romance', 'Science Fiction',
                         'TV Movie', 'Thriller', 'War', 'Western', 'Foreign']
            
            genres = st.multiselect("Select Genres", 
                                  options=all_genres,
                                  default=['Action'])
        #Feature processing
        if st.button("Predict Revenue"):
            st.header("2. Feature Processing")
            
            # Budget processing
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write("Original Budget:")
                st.write(f"${budget:,.2f}")
            
            budget_logged = np.log1p(budget)
            with col2:
                st.write("Log Transformed:")
                st.write(f"{budget_logged:.2f}")
            
            budget_scaled = scaler.transform([[budget_logged]])[0][0]
            with col3:
                st.write("Scaled Budget:")
                st.write(f"{budget_scaled:.2f}")

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
            # 4. Make Prediction
            st.header("3. Prediction Results")
            
            predicted_revenue = model.predict(input_df)[0]
            predicted_revenue = scaler_y.inverse_transform([[predicted_revenue]])[0][0]
            final_revenue = budget * (1.5 + predicted_revenue)
            
            profit_loss = final_revenue - budget
            roi = (profit_loss / budget) * 100 if budget > 0 else 0
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Revenue", f"${final_revenue:,.2f}")
            with col2:
                st.metric("Profit/Loss", f"${profit_loss:,.2f}")
            with col3:
                st.metric("ROI", f"{roi:.1f}%")
            
            # 5. Model Performance Metrics
            st.header("4. Model Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("R² Score", "0.150")
            with col2:
                st.metric("MAE", "$0.89")
            with col3:
                st.metric("RMSE", "$1.06")
            
            # 6. Visualizations
            st.header("5. Model Visualizations")
            
            # Feature Importance Plot
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
            
            # Sample Scatter Plot
            st.subheader("Actual vs Predicted Revenue")
            fig, ax = plt.subplots(figsize=(10, 6))
            plt.scatter(y_test_final, y_pred, alpha=0.5)
            plt.plot([y_test_final.min(), y_test_final.max()], 
                    [y_test_final.min(), y_test_final.max()], 
                    'r--', lw=2)
            plt.xlabel('Actual Revenue')
            plt.ylabel('Predicted Revenue')
            correlation = np.corrcoef(y_test_final.squeeze(), y_pred)[0,1]
            plt.text(0.05, 0.95, f'Correlation: {correlation:.2f}', 
                    transform=plt.gca().transAxes)
            st.pyplot(fig)
            plt.close()
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug info:")
        st.write(f"Current working directory: {os.getcwd()}")

if __name__ == "__main__":
    page_predictor_pipeline_body()
       


