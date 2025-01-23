# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Film Hit Prediction

## Note to the assessment team
"This project is my final submission for the Full Stack Software Development diploma, where I have built a data-driven application using Streamlit. Learning everything from scratch and making this project in just six weeks — without tutor support and with limited mentorship or as in my case damaging mentorship (* I will add a specific note regarding that to the submission, to better explain what happened and how it effected this project.")—made the journey incredibly challenging. However, it has been equally rewarding to create an app that holds real value for me and my colleagues in the film industry. Exploring the data has been fascinating, revealing unexpected correlations and insights that I hadn't considered before. Although time constraints and technical limitations meant I couldn't include more complex data and processing, I am excited to continue developing this app in the future with enhanced features and deeper analysis."

![Alt Text](./images/your-image.png)


You can find the deployed site here.

## Dataset Content
The dataset is sourced from Kaggle. We then created a user story based on the needs of an film investment company, 08Industries, where predictive analytics could be applied. 

The dataset has data on 5000 films in tow sets, one with 4 rows represeting movies id, title, cast, crew. The other dataset has 20 rows and represent budget, genres, homepage, id, keywords, original_language, original_title, overview, popularity, production_companies, production_countries, release_date, revenue, runtime, spoken_languages, status, tagline, title, vote_average, vote_count . In combination they are all the film data. Theweakness of the dataset is that it last was updated 8 years ago so the data is not up to date.  


## Business Requirements

08 Indusrties, an film investment company wants to invest in a film. As investment in films happens prior in greenlight, 08 Industries wanted to apply predictive analytics to try and assess the safety of their investmnet.

1 - 08 Industires are interested in discovering how the variables of the project known before greenlight correlate to the film revenue, to assess the safety of investing in films.

2 - 08 Industries is interested in predicting the revenue and profit from the variables of the film, avaialble prior to the greenlight. 

## User Stories

---

#### User story 1 - Create Dashboard (BR1,BR2)

* As a User I want to be able to access an easy-to-use dashboard that showcases graphs and other relevant information. The dashboard is divided up into 5 different pages, Project Summary, Feature Correlation, Revenue Predictor, Project Hypotheses and Validation and ML Movie Revenue Preditiction Pipeline. You can find a detailed outline of the pages below.

#### User Story 2 - Dataset Access (BR1, BR2)

* As a Software Developer, I aim to obtain a dataset that will enable me to build this project and perform essential tasks related to data manipulation and visualization. The dataset originates from Kaggle.

#### User Story 3 - Data Cleaning (BR1,BR2)

* As a Software Developer, I must clean the data in order to handle missing values and remove errors and inconsistencies. I also need to encode the string data and filter out the irrelevant data to the requiered parameters prior to greenlighting, so that I have a dataset to work with for the project. 

#### User Story 4 - Automated Data Loading (BR1, BR2)

* As a Software Developer, I plan to utilize automated data loading by importing CSV files into my Jupyter Notebooks. This approach is beneficial when working with multiple datasets or needing to load data repeatedly.

#### User Story 5 - Data Visualization (BR1)
* As a User, I want data to be represented through graphs, scatterplots, and other visualizations. This will enhance understanding and make the information more visually appealing for the client.

#### User Story 6 - Cross Industry Standard Process for Data Mining (CRISP-DM) (BR1, BR2)

* As a Software Developer, I will follow the CRISP-DM methodology to guide me through each stage of the project, from business understanding to deployment.

#### User Story 7 - Validate Hypothesis (BR1, BR2)

* As a User, I want to understand the project hypotheses and the methods used to validate them.

#### User Story 8 - Feature Engineering (BR1, BR2)

* As a Software Developer, I want to perform Feature Engineering on the data to transform raw data into a format that enhances the ability of machine learning models to recognize patterns, relationships, and trends.

#### User Story 9 - Business Requirements (BR1, BR2)

* As a Client, I want to ensure that the established Business Requirements are fulfilled and that the implemented features function correctly to meet those requirements.

#### User Story 10 - Deployment (BR1, BR2)

* As a Software Developer, I must deploy the website to Heroku and ensure that it runs smoothly without any errors.

---

## Hypothesis and how to validate?

* We suspect that the variables available prior to greenlighting are insufficient to precisely determine a film's success, making it a risky investment.

* We will validate this through a thorough correlation study between revenue and the variables available before greenlighting.

* We will validate the model's predictability by comparing its predictions against actual outcomes in a holdout validation set, calculating key performance metrics such as accuracy, precision, recall, and mean squared error (MSE) for regression tasks or confusion matrix analysis for classification tasks.  

![Alt Text](./images/your-image.png)

## The rationale to map the business requirements to the Data Visualizations and ML tasks

#### Business Requirement 1: Data Visualization and Correlation Study

* We will import, examine, preprocess, engineer features, and assess the data associated with the film project under consideration.

* We will perform an analysis to explore the relationships between variables and their influence on film revenue.

* We will utilize graphical data visualizations to confirm hypotheses and address our business requirements.

For more information, please visit the "Film Success Study" notebook.


#### Business Requirement 2: Regression Pipeline

* We want to be able to predict the revenue and profit for a film project before greenlight.

* We built and trained a regression model to help predict the revenue and profit. 

* We want to find out R2 score and Mean Absolute Error, Root Mean Squared Error, Mean Absolute Percentage Error. (See screenshots).

![Alt Text](./images/your-image.png)

### Machine Learning Business Case

* We need to implement an ML model to predict the revenue and profit of a film. For this particular project, we went with a Regression Model. A Regression Model can be defined as a model which describes the relationship between one or more independent variables and a target variable. The target variable in this case is the Revenue.

* We want to provide the client with an app that with will predict the revenue of a film prior to greenlight.  Which will give them a better chance to make a safe investment.

* The model success metrics are as follows: At least 0.75 R2 score on train and test set and the model would be considered a failure if after 12 months has elapsed, that the model would stop predicting the revenue accurately.

* As mentioned above, the target variable is revenue.


## Dashboard Design
* The dashboard contains a very simple layout. It contains 5 pages, Project Summary, Feature Correlation, Sale Price Predictor, Project Hypothesis and Validation and ML Sale Price Pipeline.

* Predictor pipeline, is divided into 4 pages: Pipeline Overview, Data Cleaning Pipeline, Feature Engineering,
Cast & Crew Engineering Pipeline.



## Unfixed Bugs
* No bugs


## Deployment
### Heroku

Create a Heroku accout if you haven't done so already.

Create a new app, give it a unique name and select your region from the options provided.

Connect to GitHub (you might be asked to confirm login through the mobile app if you have it downloaded).

Select the appropriate branch from which you want to deploy the project from.

Deploy the project. Keep an eye on the build log if the deployment fails, this will suggest any changes that need to be made in order to deploy successfully.


## Main Data Analysis and Machine Learning Libraries

Streamlit: This is what we used to create the dashboard. Building apps using Streamlit is effortless. It's particularly useful for data applications and machine learning models. It's also handy for people with minimal front-end experience.

NumPy: This is what we used to process arrays and the store the values. This is a powerful library for numerical computing in Python.

Pandas: We used this for data manipulation and analysis. It provides two main structures, Series and DataFrames. DataFrames are something we used a lot in this project.

Matplotlib: This was used for generating graphs for data visualization.

Scikit-learn: This was used for pipeline creation and the application of algorithms throughout my project. It also provides tools for predictive modelling and model evaluation.

Seaborn: This was particularly useful for the visualization of data on Streamlit. It implements attractive and styled visuals.

Git: This was used for version control. You can write a commit message using the following commands: -> **"git add ." -> "git commit -m message" -> "git push". It's good practice to keep commit messages under 50 characters.

Feature-engine: This library is essential for the Feature Engineering notebook. It also offers transformer classes.

Kaggle: This is where the dataset we used for Heritage Housing Issues was used. The link can be found above and was granted to us by Code Institute.

Python: The main programming language used for this project.


## Credits 

Coding envirment for dyslectics.

My mentor Mo Shami, for his patience and guidance through the development of this project.

My brother Sean, for his help and input.

The Churnometer walkthrough was very helpful with this project.

The template for the project is provided by Code Institute.

The Code Institute LMS contained all the lessons which helped me become familiar with the concepts around this project.

The dataset was accessed on Kaggle through Code Institute.

Roman and John of Code Institute who were very helpful and patient with my queries on the Slack channel.



## Acknowledgements (optional)
* Thank the people that provided support through this project.

