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

* We will load, inspect, clean, feature engineer and evaluate the data related to the film project in question.

* We will conduct a correlation study to better understand how each variable correlates and impacts the film revenue.

* We will use the visual representations of the data (graphs) to validate hypotheses and also anwser our business requirements.

For more information, please visit the "Film Success Study" workbook.


#### Business Requirement 2: Regression Pipeline

* We want to be able to predict the revenue and profit for the project before the greenlight.

* We built a regression model to help predict the revenue and profit. We also trained this model.

* We also want to find out R2 score and Mean Absolute Error. (See screenshots).

## ML Business Case
* In the previous bullet, you potentially visualized an ML task to answer a business requirement. You should frame the business case using the method we covered in the course 


## Dashboard Design
* List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other item that your dashboard library supports.
* Later, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project you were confident you would use a given plot to display an insight but subsequently you used another plot type).



## Unfixed Bugs
* You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable to consider, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Deployment
### Heroku

* The App live link is: https://YOUR_APP_NAME.herokuapp.com/ 
* Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.


## Main Data Analysis and Machine Learning Libraries
* Here you should list the libraries you used in the project and provide an example(s) of how you used these libraries.


## Credits 

* In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism. 
* You can break the credits section up into Content and Media, depending on what you have included in your project. 

### Content 

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open-Source site
- The images used for the gallery page were taken from this other open-source site



## Acknowledgements (optional)
* Thank the people that provided support through this project.

