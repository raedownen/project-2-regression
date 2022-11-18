# project-2-regression
Codeup Data Science Project 2 Regression

1.	Readme (.md)
•	Project Goals & Objectives
Goals
•	Construct an ML Regression model that predicts property tax assessed values ('taxvaluedollarcnt') of Single-Family Properties using attributes of the properties.
•	Find the key drivers of property value for single family properties. Some questions that come to mind are: Why do some properties have a much higher value than others when they are located so close to each other? Why are some properties valued so differently from others when they have nearly the same physical attributes but only differ in location? Is having 1 bathroom worse than having 2 bedrooms?
•	Deliver a report that the data science team can read through and replicate, understand what steps were taken, why and what the outcome was.
•	Make recommendations on what works or doesn't work in prediction these homes' values. 
Objectives
•	Document code, process (data acquisition, preparation, exploratory data analysis and statistical testing, modeling, and model evaluation), findings, and key takeaways in a jupyter notebook final report.
•	Create modules (acquire.py, prepare.py) that make your process repeatable and your report (notebook) easier to read and follow.
•	Ask exploratory questions of your data that will help you understand more about the attributes and drivers of home value. Answer questions through charts and statistical tests.
•	Construct a model to predict assessed home value for single family properties using regression techniques.
•	Make recommendations to a data science team about how to improve predictions.
•	Refine your work into a report, in the form of a jupyter notebook, that you will walk through in a 5-minute presentation to a group of colleagues and managers about your goals, the work you did, why, what you found, your methodologies, and your conclusions.
•	Be prepared to answer panel questions about your code, process, findings and key takeaways, and model.
•	Project description- Scenario-You are a junior data scientist on the Zillow data science team and receive the following email in your inbox:
We want to be able to predict the property tax assessed values ('taxvaluedollarcnt') of Single-Family Properties that had a transaction during 2017.
We have a model already, but we are hoping your insights can help us improve it. I need recommendations on a way to make a better model. Maybe you will create a new feature out of existing ones that works better, try a non-linear regression algorithm, or try to create a different model for each county. Whatever you find that works (or doesn't work) will be useful. Given you have just joined our team, we are excited to see your outside perspective.
One last thing, Maggie lost the email that told us where these properties were located. Ugh, Maggie :-/. Because property taxes are assessed at the county level, we would like to know what states and counties these are located in.
-- The Zillow Data Science Team
•	Project planning (lay out your process through the data science pipeline)
•	 Planning-I reviewed the data using MySQL to determine what columns should I use and on what key I needed to join the 3 tables.  
•	Acquisition- I used properties that had a transaction in 2017.  I noticed that there was an observation that needed to be deleted because the transaction was in 2018. So, I acquired 52441observations instead of 52442.  I used the 3 tables, properties_2017, predictions_2017, and propertylandusetype, to bring in single family residences.
•	Preparation- I used only square feet of the home, number of bedrooms, and number of bathrooms to estimate the property's assessed value, taxvaluedollarcnt, and fips. I removed fields that would cause data leakage. I removed observations where there was a null or zero for either bedrooms or bathrooms.  
•	Exploration and Pre-processing
•	Modeling
•	Delivery
•	Initial hypotheses and/or questions you have of the data, ideas
•	Data dictionary
bathroomcnt- Number of bathrooms in home including fractional bathrooms 
bedroomcnt- Number of bedrooms in home
calculatedfinishedsquarefeet- Calculated total finished living area of the home
taxvaluedollarcnt- The total tax assessed value of the parcel
	fullbathcnt- Number of full bathrooms present in the home
propertylandusetypeid- Type of land use the property is zoned for: 261 is single family residence
fips- Federal Information Processing Standard code
	the 3 counties represented are Orange, LA, and Ventura
parcelid- Unique identifier for parcels (lots)

•	Instructions or an explanation of how someone else can reproduce your project and findings (What would someone need to be able to recreate your project on their own?)
•	Key findings, recommendations, and takeaways from your project.
DATA SCIENCE PIPELINE

Planning
The goal of this stage is to clearly define your goal(s), measures of success, and plans on how to achieve that.
The deliverable is documentation of your goal, your measure of success, and how you plan on getting there. If you haven't clearly defined success, you will not know when you have achieved it.
Deliverables-Remember that you are communicating to the Zillow team, not to your instructors. So, what does the team expect to receive from you? They expect:
How to get there: You can get there by answering questions about the final product & formulating or identifying any initial hypotheses (from you or others).
Common questions include:
•	How do I determine what properties had transactions
•	What will the end product look like? Deliverables and Live Presentation
•	What format will it be in? Jupyter notebooks and slide show
•	Who will it be delivered to? Audience
•	Your customer/end user is the Zillow Data Science Team. In your deliverables, be sure to re-state your goals, as if you were delivering this to Zillow. They have asked for something from you, and you are basically communicating in a more concise way, and very clearly, the goals as you understand them and as you have taken and acted upon them through your research.
•	How will it be used? Estimating Home Value
•	How will I know I'm done? When I have completed everything in the rubric
•	What is my MVP? The basics-Deliverables and Live Presentation
•	How will I know it's good enough? Because I will do it thoroughly and I will feel confident that all criteria are met.
Formulating hypotheses
•	H0- The target variable, property tax assessed values ('taxvaluedollarcnt'), of Single Family Properties that had a transaction during 2017 will not be predictable (attributable to any influencing factors) using variables in the dataset.
•	HA- The target variable, property tax assessed values ('taxvaluedollarcnt'), of Single Family Properties that had a transaction during 2017 will be predictable (attributable to any influencing factors) using variables in the dataset.

Acquisition
AKA Data Gathering, Data Import, Data Wrangling (Acquisition + Prep)
The goal is to create a path from original data sources to the environment in which you will work with the data. You will gather data from sources to prepare and clean it in the next step. Many data scientists agree that this stage, along with the preparation stage, is where you will spend approximately 70-80% of your time.
The deliverable is a file, acquire.py, that contains the function(s) needed to reproduce the acquisition of data.
How to get there:
•	If the data source is SQL, you may need to do some clean-up, integration, aggregation or other manipulation of data in the SQL environment before reading the data into your python environment.
•	Using the Python library pandas, acquire the data into a dataframe using a function that reads from your source type, such as pandas.read_csv for acquiring data from a csv.

Preparation
AKA Data Tidying, Data Cleansing, Data Wrangling (Acquisition + Prep)
The goal is to have data, split into 3 samples (train, validate, and test), in a format that can easily be explored, analyzed and visualized. The data is split so that we have a sample we can use to test our final model, one that was not used in the exploration of the data or the development of the model. This helps us understand the generality of the model.
The deliverable is a file, prep.py, that contains the function(s) needed to reproduce the preparation of the data. The resulting dataframes should be 3 samples, a dataframe for training the algorithms, a dataframe for validating the models developed on unseen data, a dataframe for testing the best performing model to ensure the model is able to be generalized on a final set of unseen data not 'overfitting' train, a validate and a test, roughly a 70%:20%:10% split (or somewhere between that and 50%:30%:20% ... depends on amount of data available).
•	The train dataset is for training our models. We also perform our exploratory data analysis on train.
•	The validate dataset serves two purposes. First, it is an "out of sample" dataset so that we can evaluate our models on unseen data to measure how well the model generalizes. Second, the validate set allows us to fine tune our hyperparameters.
•	The test dataset is our final out of sample dataset used to evaluate how well the models tuned on validate generalize on unseen data.
How to get there:
•	Python libraries: pandas, matplotlib, seaborn, scikit-learn.
•	Use pandas to perform tasks such as handling null values, outliers, normalizing text, binning of data, changing data types, etc.
•	Use matplotlib or seaborn to plot distributions of numeric attributes and target.
•	Use scikit-learn to split the data into train and test samples.
Exploration and Pre-processing
AKA Exploratory Analysis/visualization, Feature Engineering, Feature Selection
The goal is to discover features that have the largest impact on the target variable, i.e. provide the most information gain, drive the outcome.
The deliverable is a file, preprocess.py, that contains the function(s) needed to reproduce the pre-processing of the data. The dataframe resulting from these functions should be one that is pre-processed, i.e. ready to be used in modeling. This means that attributes are reduced to features, features are in a numeric form, there are no missing values, and continuous and/or ordered values are scaled to be unitless.
How to get there:
•	Use python libraries: pandas, statsmodels, scipy, numpy, matplotlib, seaborn, scikit-learn.
•	Perform statistical testing to understand correlations, significant differences in variables, variable interdependencies, etc.
•	Create visualizations that demonstrate relationships across and within attributes and target.
•	Use domain knowledge and/or information gained through exploration to construct new features.
•	Remove features that are noisy, provide no valuable or new information, or are redundant.
•	Use scikit-learn's preprocessing algorithms (feature selection, feature engineering, dummy variables, binning, clustering, e.g.) to turn attributes into features.
Modeling
The goal is to create a robust and generalizable model that is a mapping between features and a target outcome.
The deliverable is a file, model.py, that contains functions for training the model (fit), predicting the target on new data, and evaluating results.
How to get there:
•	Python libraries: scikit-learn
•	Identify regression, classification, cross validation, and/or other algorithms that are most appropriate.
•	Build your model:
•	Create the model object.
•	Fit the model to your training, or in-sample, observations.
•	Predict the target value on your training observations.
•	Evaluate results on the in-sample predictions.
•	Repeat as necessary with other algorithms or hyperparameters.
•	Using the best performing model, predict on test, out-of-sample, observations.
•	Evaluate results on the out-of-sample predictions.
Delivery
The goal is to enable others to use what you have learned or developed through all the previous stages.
The deliverable could be of various types:
•	A pipeline.py file that takes new observations from acquisition to prediction using the previously built functions.
•	A fully deployed model.
•	A reproducible report and/or presentation with recommendations of actions to take based on original project goals.
•	Predictions made on a specific set of observations.
•	A dashboard for observing/monitoring the key drivers, or features, of the target variable.
How to get there:
•	Python sklearn's pipeline method.
•	Tableau for creating a report, presentation, story, or dashboard.
•	Jupyter notebook for creating a report or a framework to reproduce your research, e.g.
•	Flask to build a web server that provides a gateway to our model's predictions.
