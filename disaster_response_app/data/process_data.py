import sys

import pandas as pd
import re
import sqlite3

def load_data(messages_filepath, categories_filepath):
    """Load message and category data from files and merge.

    Args:
        messages_filepath (str): path to message data csv file.
        categories_filepath (str): path to categories data csv file
    
    Returns:
        merged (pandas DataFrame): dataframe of message and category data merged together
    """

    message_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    merged = message_df.merge(categories_df, how='inner', on='id')

    return merged

def clean_data(df):
    """Clean the message/category data.

    Args:
        df (pandas DataFrame): dataframe of message and category data merged together

    Returns:
        clean_cats (pandas DataFrame): dataframe of cleaned message and category data.
    """
    
    # split the categories column on semicolon and expand each split into 
    # a column of a new dataframe
    cats_expanded = df.categories.str.split(';', expand=True)

    # take the first row of the new df, split on hyphen and grab the first column of the
    # new expanded df. This column should contain the label name
    labels = list(cats_expanded.iloc[0, :].str.split('-', expand=True)[0])
    
    # remove none numerics from cats_expanded df columns - replae label names with '', leaving
    # only numeric values
    cats_expanded = cats_expanded.apply(lambda col: col.str.replace('[^0-9]', ''))
    
    # convert to int from str
    cats_expanded = cats_expanded.astype(int)
    
    # assign label names
    cats_expanded.columns = labels

    # make sure all label values are binary
    cats_expanded.where(cats_expanded < 2, 1, inplace=True)

    # merge with orginal dataframe on index and drop now defunct 'categories' column
    clean_cats = df \
        .merge(cats_expanded, how='inner', left_index=True, right_index=True) \
        .drop('categories', axis=1)

    return clean_cats

def save_data(df, database_filename):
    """Connect to/initialize sqlite3 database and write clean messages
    dataframe to database.

    Args:
        df (pandas DataFrame): dataframe of cleaned message and category data
        database_filename (str): path and name of database you wish to save cleaned 
            data to.
    
    Returns:
        None
    """

    with sqlite3.connect(database_filename) as cxn:
        df.to_sql('CleanMessages', con=cxn, if_exists='replace', index=False)

def main():
    """Entry point of program to load, clean, and save data.
    Expects command line arguments:
        messages_filepath (str): Location of csv file containing message data
        categories_filepath (str): Location of csv file containing message category data
        database_filepath (str): Location to which you wish to save cleaned messages database file
    """

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('messages_filepath', help='The location of the csv file containing your messages.')
    parser.add_argument('categories_filepath', help='The location of the csv file containing your message categories.')
    parser.add_argument('database_filepath', help='The location to which you wish to save the cleaned messages database file.')
    
    # parse args or raise expection on failure
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

    # assign command line args to vars
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