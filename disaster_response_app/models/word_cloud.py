
import wordcloud
from sqlalchemy import create_engine
import pickle
import pandas as pd 

from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from message_tokenizer import MessageTokenizer


def main():

    engine = create_engine('sqlite:///data/DisasterMessages.db')
    df = pd.read_sql_table('CleanMessages', engine)

    # word cloud data 
    # transform text messages 
    all_joined_words = ' '.join(MessageTokenizer().transform(df.message))
    all_joined_words.replace(r'[^a-zA-Z', '')
    all_joined_words.replace('http', '')
    # word_weight = Counter(all_joined_words.split(' '))
    wc = wordcloud.WordCloud(width=1000, height=1000, margin=1, max_words=150).generate(all_joined_words)

    with open('models/word_cloud.pkl', 'wb') as pkl_path:
        pickle.dump(wc, pkl_path)

if __name__ == '__main__':
    main()