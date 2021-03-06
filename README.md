# Disaster Response Pipeline Project

### Table of Contents

1. [Project Overview](#overview)
2. [Project Components](#components)
3. [Installation](#installation)
4. [File Descriptions](#files)
5. [Instructions](#instructions)
6. [Results](#results)
7. [Screenshots](#screenshots)
8. [Notes on Class Imbalance](#class_imbalance)
9. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Overview<a name="overview"></a>

Following a disaster(e.g. an earthquake or hurricane), there are a number of different problems that may arise. Different types of disaster response organizations take care of different parts of the disasters and observe messages to understand the needs of the situation. They have the least capacity to filter out messages during a large disaster, so predictive modeling can help classify different messages more efficiently.

In this project, software engineering & data engineering skills were applied to analyze disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. 

The data set contains real messages that were sent during disaster events. This real life disaster data is fed into ETL pipeline and then machine learning pipeline to categorize these events so that messages could be forwarded to an appropriate disaster relief agency. Finally, this project includes a Flask web app where an emergency worker can input a new message and get classification results in several categories i.e. The app uses a ML model to categorize any new messages received. The web app also display visualizations derived from the cleaned data.

The aim of the project is to build a Natural Language Processing tool that categorize messages using Data pipelines(ETL & ML) and then using the tool over an API.

## Project Components<a name="components"></a>

The Project is divided in the following three components:

1. ETL Pipeline - process_data.py, a data cleaning pipeline, which does the following:
   - Loads the messages and categories datasets
   - Merges the two datasets
   - Cleans the data
   - Stores it in a SQLite database

2. ML Pipeline - train_classifier.py, a machine learning pipeline, which does the following:

   - Loads data from the SQLite database
   -  Splits the dataset into training and test sets
   - Builds a text processing and machine learning pipeline
   - Trains and tunes a model using GridSearchCV
   - Outputs results on the test set
   - Exports the final model as a pickle file

3. Flask Web App - run.py, a Python script which uses Flask, which does the following; 
   - Take in file paths for database and model and display the results
   - Add data visualizations using Plotly in the web app

## Installation<a name="installation"></a>

* Python 3.5+ (I used Python 3.6)
* Data Processing & Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

The following packages also need to be installed for nltk:

  - punkt
  - wordnet
  - averaged_perceptron_tagger
  - stopwords

## File Descriptions<a name="files"></a>

There are three main foleders:
1. data
    - disaster_categories.csv: dataset including all the categories 
    - disaster_messages.csv: dataset including all the messages
    - process_data.py: ETL pipeline scripts to read, clean, and save data into a database
    - DisasterResponse.db: output of the ETL pipeline, i.e. SQLite database containing messages and categories data
2. models
    - train_classifier.py: machine learning pipeline scripts to train and export a classifier
    - classifier.pkl: output of the machine learning pipeline, i.e. a trained classifer
3. app
    - run.py: Flask file to run the web application
    - templates contains html files for the web applicatin

## Instructions<a name="instructions"></a>
### ***Run process_data.py***
 - To run ETL pipeline that cleans data and stores it in a database
 - Go to the project's root directory and run the following command:
   `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`

### ***Run train_classifier.py***
 - To run ML pipeline that trains classifier and saves it to the disk as a pickle file
 - Go to the project's root directory and run the following command:
   `python models/train_classifier.py models/DisasterResponse.db models/classifier.pkl`

### ***Run the web app***
 - To run Flask web app over an API that takes input of new disaster message and provide classification results.
 - Go to the app's directory(as html files are located in app directory only) and run the following command:
   `python run.py`
 - Go to http://0.0.0.0:3001/ to see the API and view the data visualizations
 - Input new disaster message and click 'Classify Message' to get the classification results


## Results<a name="results"></a>
 - An ETL pipleline was built to read data from two csv files, clean data, and save data into a SQLite database.
 - A machine learning pipepline was developed to train a classifier to performs multi-output classification
   on the 36 categories in the dataset.
 - A Flask app was created to show data visualization and classify the message that user enters on the web page.


## Screenshots<a name="screenshots"></a>

***Screenshot 1: App Front Page***
![Screenshot 1](https://github.com/gauravansal/Disaster-Response-Pipelines/blob/master/screenshots/screenshot%20-%20master.png)

***Screenshot 2: App Results Page***
![Screenshot 2](https://github.com/gauravansal/Disaster-Response-Pipelines/blob/master/screenshots/screenshot%20-%20go.png)


## Notes on Class Imbalance<a name="class_imbalance"></a>

The dataset included in this project is imbalanced, with very few positive examples for several message categories (i.e. some labels like water have few examples). In some cases, the proportion of positive examples is less than 5%, or even less than 1%. In such cases, the classifier accuracy can be very high (since it tends to predict that the message does not fall into these categories), and the classifier recall (i.e. the proportion of positive examples that were correctly labelled) can tend to be very low. As a result, overall f1 score of the classifier was also taken into account along with overall accuracy of the classifier.
Refer below links:
* [F1 Score](https://en.wikipedia.org/wiki/F1_score) - The F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.
* The F-score has been widely used in the natural language processing literature, such as the evaluation of [named entity recognition](https://en.wikipedia.org/wiki/Named_entity_recognition) and [word segmentation](https://en.wikipedia.org/wiki/Word_segmentation).
* The F-score is often used in the field of information retrieval for measuring search, document classification, and query classification performance. Refer below link - 
[On Understanding and Classifying Web Queries](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.127.634)


## Licensing, Authors, and Acknowledgements<a name="licensing"></a>

<a name="license"></a>
### License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<a name="acknowledgement"></a>
### Acknowledgements

This app was completed as part of the [Udacity Data Scientist Nanodegree](https://www.udacity.com/course/data-scientist-nanodegree--nd025). Starter code templates and data were provided by Udacity. The data was originally sourced by Udacity from [Figure Eight](https://www.figure-eight.com/).
















