import json
import re
import plotly
import pandas as pd
import numpy as np

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

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

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    # Bar plot of genre counts 
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # Pie chart of all the labels in the dataset
    y = np.sum(df.drop(['message', 'original', 'genre'], axis=1), axis=0)
    labels = [x for x in y.index]
    values = [x for x in y]
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Pie(
                    labels = labels,
                    values = values
                )
            ],
            'layout': {
                'title': "Distribution of Classifications in sample data"
            }
        }
                
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()