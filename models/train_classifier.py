"""
TRAIN CLASSIFIER
Disaster Resoponse Project
Udacity - Data Science Nanodegree
How to run this script (Example)
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Arguments:
    1) SQLite db path (containing pre-processed data)
    2) pickle file name to save trained ML model
"""


# import libraries
import pandas as pd
import numpy as np
import os
import pickle
import bz2
from sqlalchemy import create_engine
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score, fbeta_score, classification_report
from scipy.stats import hmean
from scipy.stats.mstats import gmean
import time
import datetime
import sys

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])


# load data from database
def load_data(database_filepath):
    """
    Load Data Function
    
    Args:
    database_filepath - path to SQLite db

    Returns:
    X - Series, Series containing features
    Y - Dataframe, dataframe containing labels i.e. categories
    category_names - list, list containing category names used for data visualization (app)
    genre_names - list, list containing genre_names used for data visualization (app)
    """
    engine = create_engine('sqlite:///'+database_filepath)
    # 'messages_categories' is the name of the table in database DisasterResponse.db
    df = pd.read_sql_table('messages_categories',engine)
    X = df['message']
    Y = df.iloc[:,3:-2]
    category_names = Y.columns[3:-2]
    genre_names = Y.columns[-2:]
    return X, Y, category_names, genre_names


# Define function tokenize to normalize, tokenize and lemmatize text string
def tokenize(text):
    """Normalize, tokenize and lemmatize text string
    
    Args:
    text: string, String containing message for processing
       
    Returns:
    clean_tokens: list, List containing normalized and lemmatized word tokens
    """

    # Replace URL links in text string with string 'urlplaceholder'
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Substitute characters in text string which match regular expression r'[^a-zA-Z0-9]'
    # with single whitespace
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    
    # Get word tokens from text string
    tokens = word_tokenize(text)
    
    # Instantiate WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()

    # Get stop words in 'English' language
    stop_words = stopwords.words("english")

    # Clean tokens
    clean_tokens = []
    for tok in tokens:
        # convert token to lowercase as stop words are in lowercase
        tok_low = tok.lower() 
        if tok_low not in stop_words:
            # Lemmatize token and remove the leading and trailing spaces from lemmatized token
            clean_tok = lemmatizer.lemmatize(tok_low).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


# Add custom Estimator
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Starting Verb Extractor class
    
    This class extract the starting verb of a sentence,
    creating a new feature for the ML classifier
    """

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if pos_tags:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return float(True)
        return float(False)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# Define custom scorer function to calculate the multi-label f-score
def multi_label_fscore(y_true, y_pred, beta=1):
    """Custom scorer function to calculate individual weighted average fbeta score of each category and
    geometric mean of weighted average fbeta score of each category
    
    Args:
    y_true: dataframe, dataframe containing true labels i.e. Y_test
    y_pred: ndarray, ndarray containing predicted labels i.e. Y_pred
    beta: numeric, beta value
       
    Returns:
    f_score_gmean: float, geometric mean of fbeta score for each category
    """
    
    b = beta
    f_score_dict = {}
    score_list = []
    
    # Create dataframe y_pred_df from ndarray y_pred 
    y_pred_df = pd.DataFrame(y_pred, columns=y_true.columns)

    for column in y_true.columns:
        score = round(fbeta_score(y_true[column], y_pred_df[column], beta, average='weighted'),4)
        score_list.append(score)
    f_score_dict['category'] = y_true.columns.tolist()
    f_score_dict['f_score'] = score_list

   
    f_score_df = pd.DataFrame.from_dict(f_score_dict)
#    print(f_score_df)

    f_score_gmean = gmean(f_score_df['f_score'])

    return f_score_gmean


# Build a machine learning model
def build_model():
    """
    Build a machine learning model
    
    This function output is a GridSearchCV object that process text messages
    according to NLP best-practice and apply a classifier.

    Args:
    None
       
    Returns:
    cv: gridsearchcv object, Gridsearchcv object that transforms the data, creates the 
    model object and finds the optimal model parameters.
    """

    pipeline = Pipeline([('features', FeatureUnion([
                                        ('text_pipeline', Pipeline([
                                            ('vect', CountVectorizer(tokenizer=tokenize)),
                                            ('tfidf', TfidfTransformer())])),
                                        ('starting_verb', StartingVerbExtractor())])),
                         ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])

    # Specify parameters for grid search
    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1,2)],
        'features__text_pipeline__vect__max_df': [0.75],
        'features__text_pipeline__vect__max_features': [5000],
        'features__text_pipeline__tfidf__use_idf': [True],
#       'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
#       'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
#       'features__text_pipeline__vect__max_features': (None, 5000, 10000),
#       'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [200],
        'clf__estimator__min_samples_split': [4],
#       'clf__estimator__n_estimators': [50,100,200],
#        'clf__estimator__min_samples_split': [2,3,4],
        'features__transformer_weights': (
            {'text_pipeline': 1, 'starting_verb': 0.5},
#           {'text_pipeline': 0.5, 'starting_verb': 1},
#           {'text_pipeline': 0.8, 'starting_verb': 1},
        )
    }

    # Specify custom scorer
    scorer = make_scorer(multi_label_fscore,greater_is_better = True)

    # create grid search object
    model = GridSearchCV(pipeline, param_grid=parameters, scoring=scorer,verbose = 2)

    return model

# Define function to evaluate model
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate Model function
    
    This function applies GridSearchCV object to a test set and prints out
    model performance (accuracy and f1score)
    
    Arguments:
        model - Scikit GridSearchCV object
        X_test - test features
        Y_test - test labels
        category_names - label names (multi-output)
    """
    Y_pred = model.predict(X_test)

    # Print overall accuracy of model on test set
    overall_accuracy = (Y_pred == Y_test).mean().mean()
    print('\nOverall accuracy of model is: {}%'.format(round(overall_accuracy*100, 2)))

    # Print overall f_score of model on test set
    multi_f_gmean = multi_label_fscore(Y_test,Y_pred, beta = 1)
    print('\nOverall f1_score for model is: {0:.2f}%'.format(multi_f_gmean*100))

# Define function to save trained model to disk as a pickle file
def save_model(model, model_filepath):
    """
    Save Model function
    
    This function saves trained model as Pickle file, to be loaded later.
    
    Args:
    model - GridSearchCV or Scikit Pipeline object
    model_filepath - destination path to save .pkl file

    Returns:
    None
    """
    # pickle file and save the model to disk.
    filename = model_filepath
    outfile = open(filename, 'wb')
    pickle.dump(model, outfile)
    outfile.close()  

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names, genre_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()