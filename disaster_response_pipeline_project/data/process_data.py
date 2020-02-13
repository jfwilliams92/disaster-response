import sys

import pandas as pd
import re
import sqlite3

def load_data(messages_filepath, categories_filepath):
    message_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    merged = message_df.merge(categories_df, how='inner', on='id')

    return merged

def clean_data(df):
    cats_expanded = df.categories.str.split(';', expand=True)
    labels = list(cats_expanded.iloc[0, :].str.split('-', expand=True)[0])
    cats_expanded = cats_expanded.apply(lambda col: col.str.replace('[^0-9]', ''))
    cats_expanded = cats_expanded.astype(int)
    cats_expanded.columns = labels
    clean_cats = df \
        .merge(cats_expanded, how='inner', left_index=True, right_index=True) \
        .drop('categories', axis=1)

    return clean_cats


def save_data(df, database_filename):

    with sqlite3.connect(database_filename) as cxn:
        df.to_sql('CleanMessages', con=cxn, if_exists='replace', index=False)
    


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('messages_filepath', help='The location of the csv file containing your messages.')
    parser.add_argument('categories_filepath', help='The location of the csv file containing your message categories.')
    parser.add_argument('database_filepath', help='The location to which you wish to save the cleaned messages database file.')
    try:
        args = parser.parse_args()
    except:
        print('Please provide the filepaths of the messages and categories '\
            'datasets as the first and second argument respectively, as '\
            'well as the filepath of the database to save the cleaned data '\
            'to as the third argument. \n\nExample: python process_data.py '\
            'disaster_messages.csv disaster_categories.csv '\
            'DisasterResponse.db')
        raise

    messages_filepath = args.messages_filepath
    categories_filepath = args.categories_filepath
    database_filepath = args.database_filepath


    print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
        .format(messages_filepath, categories_filepath))
    df = load_data(messages_filepath, categories_filepath)

    print('Cleaning data...')
    df = clean_data(df)
        
    print('Saving data...\n    DATABASE: {}'.format(database_filepath))
    save_data(df, database_filepath)
        
    print('Cleaned data saved to database!')

if __name__ == '__main__':
    main()