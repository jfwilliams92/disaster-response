
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.tokenize import word_tokenize

from sklearn.base import BaseEstimator, TransformerMixin

import re

# define a function that will allow a treebank POS tag to be converted into a WordNet
# POS Tag so the lemmatizer will understand it
def get_wordnet_pos(treebank_tag):
    """Convert a TreeBank POS tag into a WordNet POS tag.

    Args:
        treebank_tag (TreeBank POS tag): TreeBank Part of Speech tag
    
    Returns:
        wordnet tag (WordNet POS tag): WordNet Part of Speech tag
    """
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
    """Transformer that cleans and tokenizes text. Optionally removes stop words
    and lemmatizes tokens as well. Stop words are words that commonly appear in a language.
    Lemmatization is the process of reducing a word to its base form.
    """

    def __init__(self, remove_stops=True, lemmatize=True):
        """Initialize.
        
        Args:
            remove_stops (bool): Whether or not to remove stop words.
            lemmatize (bool): Whether or not to lemmatize words.
        
        Returns:
            None
        """

        self.remove_stops = remove_stops
        self.lemmatize = lemmatize
        
    def fit(self, X, y=None):
        """Placeholder method to conform with sklearn API"""

        return self
    
    def transform(self, X, y=None):
        """Clean and tokenize messages. Optionally remove stop words and lemmatize
        if self.remove_stops=True or self.lemmatize=True.
        
        Args:
            X (arr): array of text documents to transform
        
        Returns:
            X_transformed (arr): array of transformed text documents
        """

        X_transformed = []
        
        # iterate over supplied messages
        for text in X: 
            # sub out any urls 
            text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
            
            # remove all non-alphanumeric characters
            text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
              
            # lower text and strip whitespace
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