# Airline Satisfaction Machine Learning Analysis

## Project Directive
Develop and deploy a machine learning user application to predict passenger satisfaction from inputs of a flight experience survey and quantify the importance of variables impacting customer satisfaction.

## Dataset
We retrieved the dataset from Kaggle. The data consists of a training dataset (data used to train our machine learning model) and a testing dataset (data used to test the model). The dataset includes 22 variables impacting a customer’s flight experience – trip length, in-flight Wi-Fi service, online boarding, class, type of travel, etc. The data was gathered by asking random passengers from various flights/airlines to fill-out a survey after their flight, and rank these attributes on a 1-5 scale, including whether they were “satisfied” or “dissatisfied” with the flight experience overall. For the purposes of our machine learning model, we deemed survey inputs to be input variables, and the overall satisfaction or dissatisfaction the target variable for our model to predict. 

## Machine Learning Model Testing
In training and testing our machine learning model, we experimented with 3 different algorithms – logistic regression, classic decision tree, and random forest. For our final application, we employed a random forest model, as this algorithm produced the best model score, and we utilized the “feature importance” functionality for random forest models in the Sci-kit Learn Python library to quantify which of the survey inputs were most impactful on customer satisfaction.

## Deployment
We deployed the model as a web application at the following address: https://airline-satisfaction.herokuapp.com/.

## Conclusion
The utility of the application is for stakeholders, whether that be airlines, airline investors, airline travelers, etc. to test the application to determine what factors of the flight experience should be optimized to produce customer satisfaction. On the web application, we also included graphs and analyses to support our conclusions from the data analysis. 
