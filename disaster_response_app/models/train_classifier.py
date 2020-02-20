
# import libraries

import sys
import pandas as pd
import sqlite3
import re
import pickle

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# import custom MessageTokenizer class
from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from message_tokenizer import MessageTokenizer


def load_data(database_filepath):
    """Load message and category data from a sqlite3 database file.

    Args:
        database_filepath (str): location of the database file to load data from.
    
    Returns:
        X (arr): array of text documents
        Y (arr): multidimensional array of shape (n_messages, n_message_labels)
        labels (arr): list of label names
    """
    # connect to db and read table
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * FROM CleanMessages', engine)

    # define variables. X is input, Y is target
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1) 

    return X.values, Y.values, list(Y.columns)

# ignore ConvergenceWarnings
@ignore_warnings(category=ConvergenceWarning)
def build_model(X_train, Y_train, n_iter):
    """Fit an estimator on training data. Optionally tune estimator with 
    a RandomizedSearch over hyperparameter space.

    Args:
        X_train (arr): array of text documents.
        Y_train (arr): array of message labels of shape (n_messages, n_message_labels)
        n_iter (int): number of hyperparameter combinations to sample. If 0, will fit 
            model on preselected hyperparameter combination.
    
    Returns:
        fitted_pipeline: pipeline fitted on training data with tuned or selected hyperparameters.
    """
    # search for best hyperparameters if n_iter > 0
    if n_iter > 0:
        print('Searching for best hyperparameters...')

        # create base pipeline
        pipeline_log = Pipeline([
            ('msg_tokenizer', MessageTokenizer()),
            # Count Vectorizer with Tokenizer
            ('count_vec', CountVectorizer()),
            # TF-IDF Transformer
            ('tfidf', TfidfTransformer()),
            # classifier - one classifier per label
            ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear')))
        ])

        # tune the pipeline with hamming loss
        hamming_scorer = make_scorer(hamming_loss, greater_is_better=False)
        # list out all possible hyperparameter values
        search_params = {
            'msg_tokenizer__remove_stops': [False, True],
            'msg_tokenizer__lemmatize': [False, True],
            'tfidf__norm': [None, 'l1', 'l2'],
            'tfidf__use_idf': [False, True],
            'tfidf__smooth_idf': [False, True],
            'count_vec__ngram_range': [(1,1), (1,2), (1,3), (1,4)],
            'count_vec__max_features': [None, 100, 500, 1000],
            'clf__estimator__dual': [False, True],
            'clf__estimator__C': [1, 10, 50, 100],
            'clf__estimator__class_weight': [None, 'balanced']
        }
        # search over hyperparameter space with n_iter combinations selected - each selection is fit and scored cv=3 times
        cv_log = RandomizedSearchCV(pipeline_log, search_params, n_iter=n_iter, cv=3, scoring=hamming_scorer, verbose=2)
        search_log = cv_log.fit(X_train, Y_train)

        print('Best params found: \n')
        print(search_log.best_params_)

        # return the estimator/hyperparam combination that had the best score in the RandomizedSearch
        return search_log.best_estimator_

    # if n_iter = 0, fit on training data with preselected hyperparameters
    else:
        print('Fitting pipeline with pre-selected hyperparameters.')
        pipeline_log = Pipeline([
            ('msg_tokenizer', MessageTokenizer(remove_stops=False, lemmatize=True)),
            # Count Vectorizer with Tokenizer
            ('count_vec', CountVectorizer(ngram_range=(1,4), max_features=1000)),
            # TF-IDF Transformer
            ('tfidf', TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False)),
            # classifier - one classifier per label
            ('clf', OneVsRestClassifier(LogisticRegression(solver='liblinear', C=10, dual=True, class_weight=None)))
        ])

        pipeline_log.fit(X_train, Y_train)

        return pipeline_log

def evaluate_model(model, X_test, Y_test, category_names):
    """Predict on test data and print out evaluation of predictions.

    Args:
        model: trained estimator
        X_test (arr): array of text documents
        Y_test (arr): multidim array of shape (n_messages, n_message_labels)
        category_names (list): list of message label names
    
    Returns:
        None
    """
    # predict on the test data
    y_pred = model.predict(X_test)
    if isinstance(Y_test, pd.DataFrame):
        Y_test = Y_test.values
    # print out metrics
    for idx, label in enumerate(category_names):
        print('\n', f'MESSAGE LABEL: {label}', '\n')
        print("Confusion Matrix: ")
        # convert confusion matrix to dataframe to allow labeling of axes
        print(pd.DataFrame(data=confusion_matrix(Y_test[:, idx], y_pred[:, idx]), columns=['PredNeg', 'PredPos'], index=['TrueNeg', 'TruePos']))
        print('\n')
        print("Classification Report: ")
        print(classification_report(Y_test[:, idx], y_pred[:, idx]))

def save_model(model, model_filepath):
    """Save trained model to pickle file.

    Args:
        model: trained estimator
        model_filepath (str): path to store pickled model
    
    Returns:
        None
    """

    with open(model_filepath, 'wb') as pkl_path:
        pickle.dump(model, pkl_path)

def main():
    """"Main entry point of program. Takes command line arguments as defined in the ArgumentParser.
    Expected command line arguments are:
    database_filepath (str): location of databasefile containing the message data
    model_filepath (str): path to store pickled model
    n_tune_iter (int): optional, number of hyperparameter combinations to test

    Returns: 
        None
    """

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('database_filepath', help='The location of the database file containing the message data.')
    parser.add_argument('model_filepath', help='The filepath where you wish the fitted model to be saved.')
    parser.add_argument('--n_tune_iter', type=int, help='Optional: number of hyperparameter combinations to test.')
    try:
        args = parser.parse_args()
    except:
        print(
            """Argument parsing failed. Please provide the location of the database file containing the message data,
            the filepath where you wish the fitted model to be saved, and the number of 
            hyperparameter combinations you would like to test. If you do not supply --n_tune_iter, 
            then the model will be fit on preselected hyperparameters.
             \n\nExample usage: 
            python train_classifier.py ../data/DisasterResponse.db classifier.pkl --n_tune_iter 10
            """
        )
        raise
    # assign command line arguments to vars
    database_filepath, model_filepath = args.database_filepath, args.model_filepath
    # if optional n_tune_iter is not supplied, set to 0
    n_tune_iter = 0 if not args.n_tune_iter else args.n_tune_iter

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
    print('Building model and training model...')
    model = build_model(X_train, Y_train, n_iter=n_tune_iter)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')

if __name__ == '__main__':
    main()