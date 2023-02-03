import sys
import pandas as pd
import re
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Function that loads the messages and the categories datasets and joins
    them into 1 dataset

    Args:
        messages_filepath: location where the csv file of the messages dataset is saved
        categories_filepath: location where the csv file of the categories dataset is saved
    
    Output:
        df: Pandas Dataframe where messages and categories are joined
    '''
    
    # Load raw sets
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')
    
    # Join on ID
    df = messages.join(categories)
    
    return df
    

def clean_data(df):
    '''
    Function that cleans the dataset making it usable for ML-models

    Including:
    - Splitting the categories column in multiple numeric labels
    - Drop outliers
    - Drop redundant columns
    - Drop duplicate data

    Args:
        df: raw pandas Dataframe of messages + categories
    
    Output:
        df: cleaned pandas Dataframe
    '''
    cat_cols = re.split('-.;', df['categories'][2])
    cat_cols = [col.strip('-0') for col in cat_cols]
    
    df[cat_cols] = df['categories'].str.split(';', expand=True)

    for column in cat_cols:
        # set each value to be the last character of the string
        df[column] = pd.to_numeric(df[column].str.split("-").str[1])
    
    df = df.drop('categories', axis=1)
    
    # Assume related = 2 are errors + drop child alone because it never occurs
    df = df[df['related'] < 2 ]
    df = df.drop('child_alone', axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    '''
    Function that saves the dataset in a sqlite database in a table called messages

    Args:
        df: Database to save
        database_filename: (potential) location of the sqlite database
    '''
    engine = create_engine(f"sqlite:///{database_filename}")
    df.to_sql('messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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