# Disaster Response Pipeline
![Intro Pic](screenshots/main_pagepng)

## Table of Contents
1. [Project Description](#description)
2. [Getting Started](#getting_started)
	1. [Libraries](#Libraries)
	2. [Executing Program](#execution)
	3. [Additional Material](#material)
3. [Authors](#authors)
4. [License](#license)
5. [Acknowledgement](#acknowledgement)


## Project Description
The project is to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classify disaster message on a real time basis. After searching key messages in the search box, the categoriries of the event will be sent to an appropriate disaster relief agency. 

This project is divided in the following key steps: 

### 1. ETL pipeline
Load the raw csv-file provided, extract the messages and categories as a dataframe, clean the data and load it into an SQLite database.
### 2. Machine Learning pipeline
Build a machine learning pipeline using NLTK, as well as scikit-learn's pipeline and GridSearchCV, export the model to a pickle file.
### 3. Flask App
Run a web app, display results in a flash web app, creat data visualizations.

### Libraries 
- re
- Pandas
- SQLAlchemy
- Pickle
- NLTK
- SKLearn
- Flask
- Plotly



