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
    f"* This pipeline uses a Random Forest model optimized through GridSearchCV to predict movie revenues. \n"
    f"* The model achieved an R² score of 0.15, showing the challenge of predicting exact movie revenues. \n"
    f"* Budget is the most influential feature, accounting for 46.7% of prediction weight. \n"
    f"* Genre factors like Comedy (4.6%), Drama (4.4%), and Thriller (4.2%) also play significant roles. \n"
    f"* The model performs with an RMSE of $1.06 and MAE of $0.89 on logged, scaled revenue values."
    )
        

        st.write("### ML Pipeline Steps")

    # Sample input
        st.header("1. Enter Movie Details")

       
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
                    default=['Action']

        #Feature processing
        if st.button("Predict Revenue"):
            st.header("2. Feature Processing")
            
            # Budget processing
            st.write("Budget Processing:")
            st.write(f"Original Budget:${budget:,.2f}")
            
            budget_logged = np.log1p(budget)
            st.write(f"Log Transformed: {budget_logged:.2f}")
               
            budget_scaled = scaler.transform([[budget_logged]])[0][0]
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
        
        with tab2:
            st.subheader("Residual Analysis")
        # Calculate residuals
            residuals = y_test_final - y_pred
    
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
        # Scatter plot of residuals
            ax1.scatter(y_pred, residuals, alpha=0.5)
            ax1.axhline(y=0, color='r', linestyle='--')
            ax1.set_xlabel('Predicted Revenue')
            ax1.set_ylabel('Residuals')
            ax1.set_title('Residuals vs Predicted Values')
    
        # Distribution of residuals
            sns.histplot(residuals, kde=True, ax=ax2)
            ax2.set_title('Distribution of Residuals')
            ax2.set_xlabel('Residual Value')
    
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            st.subheader("Distribution of Predictions")
            fig, ax = plt.subplots(figsize=(10, 6))
    
        # Plot actual and predicted distributions
            sns.kdeplot(data=y_test_final.squeeze(), label='Actual Revenue', ax=ax)
            sns.kdeplot(data=y_pred, label='Predicted Revenue', ax=ax)
            plt.title('Distribution of Actual vs Predicted Revenue')
            plt.xlabel('Revenue')
            plt.legend()
    
        st.pyplot(fig)
        plt.close()

    with tab4:
        st.subheader("ROI Analysis")
    
    # ROI for actual and predicted values
        roi_actual = ((y_test_final - X_test_final['budget_scaled']) / X_test_final['budget_scaled']) * 100
        roi_pred = ((y_pred - X_test_final['budget_scaled']) / X_test_final['budget_scaled']) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # ROI Distribution
        sns.histplot(data=roi_pred, kde=True, ax=ax1)
        ax1.set_title('Distribution of Predicted ROI')
        ax1.set_xlabel('ROI (%)')
    
    # ROI Scatter
        ax2.scatter(roi_actual, roi_pred, alpha=0.5)
        ax2.plot([roi_actual.min(), roi_actual.max()], 
                [roi_actual.min(), roi_actual.max()], 
                'r--', lw=2)
        ax2.set_xlabel('Actual ROI (%)')
        ax2.set_ylabel('Predicted ROI (%)')
        ax2.set_title('Predicted vs Actual ROI')
    
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # explanatory text
    st.write("""
    ### Visualization Insights:
    * The Feature Importance plot shows which factors most strongly influence revenue predictions
    * The Residual Analysis helps us understand where our model might be over or under-predicting
    * The Distribution comparison shows how well our predictions match the actual revenue distribution
    * The ROI Analysis provides insights into the financial performance predictions
    """)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Debug info:")
        st.write(f"Current working directory: {os.getcwd()}")

if __name__ == "__main__":
    page_predictor_pipeline_body()
       


