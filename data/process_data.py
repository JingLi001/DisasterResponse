import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """
    Load data from messages and categories datasets
    
    Arguments:
        messages_filepath   -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> merged dataframe containing messages and categories
    """
    messages = pd.read_csv(messages_filepath,encoding='latin-1')
    categories = pd.read_csv(categories_filepath,encoding='latin-1')
    df = pd.merge(messages,categories,on='id')
    
    return df 


def clean_data(df):
    
    """
    clean data 
    
    Arguments:
        df -> Combined data containing messages and categories
    Outputs:
        df -> Combined data containing messages and categories with categories cleaned up
        
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
     # select the first row of the categories dataframe
    firstrow = categories.loc[0].str.split('-').apply(lambda x: x[0]).tolist()
     
     # rename the columns of `categories`
    categories.columns = firstrow 
        
    for column in categories:
        # set each value to be the last character of the string
       categories[column] = categories[column].str[-1]
        # convert column from string to numeric
       categories[column] = categories[column].astype(int)
        
    #drop child_alone category since it has all zeros only without other lables. (print(df.sum()))
    categories = categories.drop(['child_alone'], axis = 1)
    # related column contains some values of '2' which could be neglible, re-assign them as '1'
    categories.loc[categories['related']==2,'related']=1
    
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], join='inner', axis = 1)
    # drop duplicates
    df = df[~df.duplicated()]
    return df


def save_data(df, database_filename):
    
    """
    Save the clean dataset into an sqlite database.
    
    Arguments:
        df -> Combined data containing messages and categories with categories cleaned up
        database_filename -> Path to SQLite destination database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages', engine, index=False, if_exists='replace')
       


def main():
    if len(sys.argv) == 4:
        # Extract the parameters in relevant variable
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        
        print('Cleaning data...')
        df = clean_data(df)
       # print(df.head())
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