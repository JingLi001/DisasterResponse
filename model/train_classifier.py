# import libraries
import sys

from sqlalchemy import create_engine
from sqlalchemy import inspect
import pandas as pd
import re


import pickle

from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV




from nltk.tokenize import word_tokenize
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])



#Functions

def load_data(database_filepath):

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('Messages', engine)
    X = df['message']
    y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    return X, y, y.columns.tolist()


def tokenize(text):
    
    """
    INPUT:
    text - string
    OUTPUT:
    tokens - list of strings
    
    function takes raw text, removes punctuation signs, substitutes
    with spaces. Puts all characters in lower case, tokenizes text
    by words, removes stop words, lemmatizes, and returns list of tokens 
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
    generates an NLP model that is ready to be fit with training data
    -------
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__ngram_range': ((1, 1), (1,2)),
        'clf__estimator__n_estimators' : [50, 80]
        }
    
    cv = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    f1_scores = []
    for ind, cat in enumerate(Y_test):
        print('Class - {}'.format(cat))
        print(classification_report(Y_test.values[ind], y_pred[ind], zero_division = 1))
    
      
def save_model(model, model_filepath):
    """
    Parameters
    ----------
    model : ML model
        trained and ready to be deployed to production.
    model_filepath : string
        distination to be saved.
    """
    # save the model to disk
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))


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
    