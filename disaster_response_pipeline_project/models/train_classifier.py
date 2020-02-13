import sys

# import libraries
import pandas as pd
import sqlite3
import re

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss, make_scorer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

import pickle

# define a function that will allow a treebank POS tag to be converted into a WordNet
# POS Tag so the lemmatizer will understand it
def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    # default to Noun 
    else:
        return wordnet.NOUN

# implement a custom transformer to determine if removing stops and/or lemmatizing improves model performance
class MessageTokenizer(BaseEstimator, TransformerMixin):
    def __init__(self, remove_stops=True, lemmatize=True):
        self.remove_stops = remove_stops
        self.lemmatize = lemmatize
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_transformed = []
        
        # iterate over supplied messages
        for text in X: 
            # sub out any urls 
            text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            
            # remove all non-alphanumeric characters
            text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
              
            # lower and strip whitespace
            text = text.lower().strip()
    
            # tokenize words - nltk.tokenize.word_tokenize
            words = word_tokenize(text)
            
            if self.lemmatize:
                # tag words with Part of Speech - list of (word, POS) tuples 
                # nltk.pos_tag()
                words_with_pos_tag = pos_tag(words)
                
                if self.remove_stops:
                    # remove stop words
                    # stop_words = nlt.corpus.stopwords of 'english' language
                    words_with_pos_tag = [word for word in words_with_pos_tag if word[0] not in stop_words]
                
                # change pos tags to wordnet pos tags for lemmatizer
                words_with_wordnet_tag = []
    
                for word_with_tag in words_with_pos_tag:
                    word, tag = word_with_tag
                    tag = get_wordnet_pos(tag)
                    words_with_wordnet_tag.append((word, tag))

                # lemmatize
                lemm = WordNetLemmatizer()
                # unpack the (word, pos) tuple into the Lemmatizer to give better lemmatization 
                # lemmatization is more effective when it knows the correct part of speech
                words = [lemm.lemmatize(*w) for w in words_with_wordnet_tag]
                
            else:
                if self.remove_stops:
                    words = [word for word in words if word not in stop_words]

            # join cleaned words back into single document
            X_transformed.append(' '.join(words))
        
        return X_transformed 

def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql('SELECT * FROM CleanMessages', engine)

    # define variables. X is input, Y is target
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1) 

    # check to make sure we have at least one instance for each label
    labels_with_no_instance = Y.columns[~(Y == 1).any(axis=0)]
    Y = Y.drop(labels_with_no_instance, axis=1)

    # drop rows that are non-binary
    drop_index = Y[~Y.isin([0, 1]).all(axis=1)].index
    Y = Y.drop(drop_index)
    X = X.drop(drop_index)

    return X.values, Y.values, list(Y.columns)

# ignore ConvergenceWarnings
@ignore_warnings(category=ConvergenceWarning)
def build_model(X_train, Y_train, n_iter):
    if n_iter > 0:

        pipeline_log = Pipeline([
            ('msg_tokenizer', MessageTokenizer()),
            # Count Vectorizer with Tokenizer
            ('count_vec', CountVectorizer()),
            # TF-IDF Transformer
            ('tfidf', TfidfTransformer()),
            # classifier - one classifier per label
            ('clf', OneVsRestClassifier(LogisticRegression(solver='lbfgs')))
        ])

        # tune the grid with hamming loss
        hamming_scorer = make_scorer(hamming_loss, greater_is_better=False)
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
        cv_log = RandomizedSearchCV(pipeline_log, search_params, n_iter=n_iter, cv=3, scoring=hamming_scorer, verbose=2)
        search_log = cv_log.fit(X_train, Y_train)

        print('Best params found: \n')
        print(search_log.best_params_)

        return search_log.best_estimator_
    
    else:
        pipeline_log = Pipeline([
            ('msg_tokenizer', MessageTokenizer(remove_stops=False, lemmatize=True)),
            # Count Vectorizer with Tokenizer
            ('count_vec', CountVectorizer(ngram_range=(1,4), max_features=1000)),
            # TF-IDF Transformer
            ('tfidf', TfidfTransformer(norm='l2', use_idf=False, smooth_idf=False)),
            # classifier - one classifier per label
            ('clf', OneVsRestClassifier(LogisticRegression(solver='lbfgs', C=10, dual=True, class_weight=None)))
        ])

        pipeline_log.fit(X_train, Y_train)

        return pipeline_log

def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    if isinstance(Y_test, pd.DataFrame):
        Y_test = Y_test.values
    for idx, label in enumerate(category_names):
        print('\n', f'MESSAGE LABEL: {label}', '\n')
        print("Confusion Matrix: ")
        print(pd.DataFrame(data=confusion_matrix(Y_test[:, idx], y_pred[:, idx]), columns=['PredNeg', 'PredPos'], index=['TrueNeg', 'TruePos']))
        print('\n')
        print("Classification Report: ")
        print(classification_report(Y_test[:, idx], y_pred[:, idx]))

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as pkl_path:
        pickle.dump(model, pkl_path)

def main():
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
            hyperparameter combinations you would like to test.  \n\nExample usage: 
            python train_classifier.py ../data/DisasterResponse.db classifier.pkl 10
            """
        )
        raise
    database_filepath, model_filepath = args.database_filepath, args.model_filepath
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