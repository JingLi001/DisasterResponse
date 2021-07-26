import sys
import os
import re
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import inspect

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator,TransformerMixin


def load_data(database_filepath):
    
    """
    Load Data from the Database 
    
    Arguments:
        database_filepath -> Path to SQLite destination database (e.g. DisasterResponse.db)
    Output:
        X -> a dataframe containing features
        Y -> a dataframe containing labels
        category_names -> List of categories name
    """    
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages', engine)
   
    
    X = df['message']
    y = df.iloc[:,4:]
    
    print(X)
    #print(y.columns)
    category_names = y.columns 
    
    return X, y, category_names

def tokenize(text):
    
    """
    Tokenize the text function
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        clean_tokens -> List of tokens extracted from the text
    """
    # Replace all urls with a urlplaceholder string
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Extract all the urls from the provided text 
    detected_urls = re.findall(url_regex, text)
    for detected_url in detected_urls:
        text = text.replace(detected_url, 'urlplaceholder')
        
    # tokenize the text    
    tokens = nltk.word_tokenize(text)
    # initialize lemmatizer
    lemmatizer = nltk.WordNetLemmatizer()
     # List of clean tokens
    clean_tokens = [lemmatizer.lemmatize(tok).lower().strip() for tok in tokens]
    
    return clean_tokens


def build_model():
    
    """
    generates an NLP pipeline
    
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    # Two parameters have been selected which can be identified by the help of pipeline.get_params().keys()
    parameters = {
        'vect__ngram_range': ((1, 1), (1,2)),
        'clf__estimator__n_estimators' : [50, 80]
        }
    
 
    cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1)
    
    return cv
    


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Evaluates the above model based on the data provided
    Arguments:
        model-> the scikit fitted model
        X_test->test samples
        Y_test-> test classifications
        category_names-> names of columns of dataframe
   output
        None
    """
    y_pred = model.predict(X_test)
    
   
    
    for col in Y_test.columns:
        print(classification_report(Y_test.values, y_pred, target_names=col))
    



def save_model(model, model_filepath):
    
     """
   
    Save trained model as Pickle file.
    
    Arguments:
        model-> GridSearchCV or Scikit Pipelin object
        model_filepath -> destination path to save .pkl file
    
    """
    # save the model to disk
   
   
     pickle.dump(model, open(model_filepath, 'wb'))
   


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
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