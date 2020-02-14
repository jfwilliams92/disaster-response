import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from message_tokenizer import MessageTokenizer

app = Flask(__name__)

engine = create_engine('sqlite:///data/DisasterMessages.db')
df = pd.read_sql_table('CleanMessages', engine)

# load model
model = joblib.load("models/disaster_logit.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
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
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    # print('here at the index page!')
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 
    # classification type comes from the radio button selection
    classification_type = request.args.get('classification_type', 'hard_class')
    
    # use model to predict classification for query
    # 0 1 prediction for 'hard_class'
    if classification_type == 'hard_class':
        classification_labels = model.predict([query])[0]
    # predict label probability for 'soft_class'
    elif classification_type == 'soft_class':
        classification_labels = model.predict_proba([query])[0]
    
    # filter to labels with any positive instances
    # this is what happens during classifier training - otherwise there will be a prediction-label mismatch
    df_labels = (df.iloc[:, 4:] == 1).any(axis=0) 
    classification_keys = list(df_labels[df_labels.values].index)
    classification_results = dict(zip(classification_keys, classification_labels))

    # pass the data back to the go.html page 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        classification_type=classification_type
    )


def main():
    app.run(debug=True)

if __name__ == '__main__':
    main()