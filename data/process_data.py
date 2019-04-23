"""
PREPROCESSING DATA
Disaster Response Pipeline Project
Udacity - Data Science Nanodegree
Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
Arguments:
    1) CSV file containing messages (disaster_messages.csv)
    2) CSV file containing categories (disaster_categories.csv)
    3) SQLite destination database (DisasterResponse.db)
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
	"""Load and merge messages and categories datasets
	
	Args:
	messages_filepath: string, Filepath for csv file containing messages dataset.
	categories_filepath: string, Filepath for csv file containing categories dataset.
	   
	Returns:
	df: dataframe, Dataframe containing merged content of messages and categories datasets.
	"""

	# Load messages dataset
	messages = pd.read_csv(messages_filepath)

	# Load categories dataset
	categories = pd.read_csv(categories_filepath)

	# merge datasets
	df = messages.merge(categories, how='inner', on='id')

	return df


def clean_data(df):
	"""Clean dataframe by removing duplicates, converting categories from strings 
	to binary values, removing unrequired or improper features and converting
	categorical features into dummt features
	
	Args:
	df: dataframe, Dataframe containing merged content of messages and categories datasets.
	   
	Returns:
	df: dataframe, Dataframe containing cleaned version of input dataframe.
	"""

	# create a dataframe of the 36 individual category columns
	categories = df['categories'].str.split(';', expand=True)

	# select the first row of the categories dataframe
	row = categories.iloc[0,:]

	# use this row to extract a list of new column names for categories.
	# one way is to apply a lambda function that takes everything 
	# up to the second to last character of each string with slicing
	category_colnames = row.apply(lambda x: x[:-2])
	category_colnames = category_colnames.tolist()

	# Rename the columns of `categories`
	categories.columns = category_colnames

	# Convert  category values to numeric values 0 or 1
	for column in categories:
		# set each value to be the last character of the string
		categories[column] = categories[column].str[-1]
	
		# convert column from string to numeric
		categories[column] = pd.to_numeric(categories[column])

	# Drop column child_alone from categories dataframe.
	categories.drop('child_alone', axis = 1, inplace = True)

	# Drop the original categories column from `df`
	df.drop('categories', axis=1, inplace=True)

	# concatenate the original dataframe with the new `categories` dataframe
	df = pd.concat([df, categories], axis=1, join='inner')

	# Drop duplicates
	df.drop_duplicates(subset=None, keep='first', inplace=True)

	# Drop observations with column "related" having value as 2
	df.drop(df.index[df['related'] == 2].tolist(), axis=0, inplace=True)

	# Convert categorical feature genre into dummy variables
	df = pd.get_dummies(df, columns=['genre'], prefix='genre', prefix_sep='_', drop_first=True)

	return df


def save_data(df, database_filename):
	"""Save cleaned data into an SQLite database.
	
	Args:
	df: dataframe, Dataframe containing cleaned version of merged message and 
	categories data.
	database_filename: string, Filename for output database.
	   
	Returns:
	None
	""" 
	engine = create_engine('sqlite:///' + database_filename)
	df.to_sql('messages_categories', engine, if_exists='replace', index=False)


def main():
	if len(sys.argv) == 4:

		messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

		print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
			  .format(messages_filepath, categories_filepath))
		df = load_data(messages_filepath, categories_filepath)

		print('Cleaning data...')
		df = clean_data(df)
		
		print('Saving data...\n    DATABASE: {}'.format(database_filepath))
		save_data(df, database_filepath)
		
		print('Cleaned data saved to database!')
	
	else:
		print('Please provide the filepaths of the messages and categories '\
			  'datasets as the first and second argument respectively, as '\
			  'well as the filepath of the database to save the cleaned data '\
			  'to as the third argument. \n\nExample: python process_data.py '\
			  'disaster_messages.csv disaster_categories.csv '\
			  'DisasterResponse.db')


if __name__ == '__main__':
	main()