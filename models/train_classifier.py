import re
import sys
import nltk
import pickle
import pandas as pd
from sqlalchemy import create_engine 
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def load_data(database_filepath):
    '''
    Function that loads the dataset from the sqlite database and splits it
    into X and Y variables

    Args:
        database_filepath: location where the sqlite database is saved
    
    Output:
        X: The messages on which the predictions will be based
        Y: The labels of the messages
        category_names: The names of the different Y-labels
    '''
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('messages', engine)
    
    X = df['message']
    Y = df.drop(['message', 'original', 'genre'], axis=1)
    category_names = Y.columns
    
    return X, Y, category_names


def tokenize(text):
    '''
    Tokenizer that cleans a string by substituting anything but letters, tokenization,
    removing stopwords and stemming the words.

    Args:
        text: the string to be cleaned
    
    Output:
        text: list of words after cleaning
    '''
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    text = word_tokenize(text)
    text = [word for word in text if text not in stopwords.words("english")]
    text = [PorterStemmer().stem(word) for word in text]
    return text


def build_model():
    '''
    Function that created the model-pipeline and including a grid search to find the best parameters

    Args: 
        None
    
    Output:
        cv: a grid search object which can be fitted to the training data
    '''
    pipeline = Pipeline([
    
    ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),
    
            
     ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {
        'features__text_pipeline__vect__ngram_range': [(1,1), (1,2)],
        'clf__estimator__n_estimators': [50, 100],
        'clf__estimator__min_samples_split': [2, 3],
        'clf__estimator__criterion': ['gini', 'entropy']
    }

    cv = RandomizedSearchCV(pipeline, param_distributions=parameters, cv=2)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function that prints a classification report to see how well the model performed

    Args:
        model: The trained model
        X_test: The data to make predictions on
        Y_test: The true labels of X_test
        category_names: The names of the labels to be printed in the classification report
    
    Output:
        A print of the classification report for the predictions + labels
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.values, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    '''
    Function that saves the trained model

    Args:
        model: The trained ML-model
        model_filepath: Location to save the model to
    
    Output:
        A pickle dump of the model to the model_filepath
    '''  
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