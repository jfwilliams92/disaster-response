
import wordcloud
from sqlalchemy import create_engine
import pickle
import pandas as pd 


# import custom MessageTokenizer
from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from message_tokenizer import MessageTokenizer


def main():
    """"Create a word cloud from cleaned messages.

    Args: 
        None
    
    Returns:
        None
    """

    # connect to cleaned messages DB - /data/process_data.py must be run first
    engine = create_engine('sqlite:///data/DisasterMessages.db')
    df = pd.read_sql_table('CleanMessages', engine)

    # word cloud data 
    # transform text messages 
    all_joined_words = ' '.join(MessageTokenizer().transform(df.message))
    
    # some extra cleaning for the word cloud
    all_joined_words.replace(r'[^a-zA-Z', '')
    all_joined_words.replace('http', '')

    # create the WordCloud object with top 150 words
    wc = wordcloud.WordCloud(width=1000, height=1000, margin=1, max_words=150).generate(all_joined_words)

    # save it off
    with open('models/word_cloud.pkl', 'wb') as pkl_path:
        pickle.dump(wc, pkl_path)

if __name__ == '__main__':
    main()