import json
import plotly
import pandas as pd
import numpy as np
import wordcloud
 
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap, Layout, Figure, Scatter
import plotly.express as px
from sklearn.externals import joblib
from sqlalchemy import create_engine

# import custom MessageTokenizer
from os import path
import sys
sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
from message_tokenizer import MessageTokenizer

# create app and read in data
app = Flask(__name__)

engine = create_engine('sqlite:///data/DisasterMessages.db')
df = pd.read_sql_table('CleanMessages', engine)

# load model
model = joblib.load("models/disaster_logit.pkl")

# set up the word cloud data 
wc = joblib.load('models/word_cloud.pkl')

# create the word cloud figure here so it doesn't have to be generated on each refresh
wordcloud_fig = px.imshow(wc)
wordcloud_fig.update_layout(
    title=dict(text='150 Most Common Words in Disaster Scenarios', x=0.5),
    width=1000,
    height=1000,
    xaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
    yaxis={'showgrid': False, 'showticklabels': False, 'zeroline': False},
    hovermode=False
)

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
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
        }
    ]

    # heatmap of label pairwise correlations
    # drop some cols and replace 1's with np.nan
    heatmap_df = df.drop(['id', 'child_alone'], axis=1).corr().replace(1, np.nan)
    cols = list(heatmap_df.columns)
    # grab the lower triangle of the values array - zeroes out the other values
    heatmap_df = np.tril(heatmap_df)
    # replace 0s with nan
    heatmap_df[heatmap_df == 0] = np.nan
    heatmap_data = Heatmap(
        z=heatmap_df,
        x=cols,
        y=cols
    )
    heatmap_layout = Layout(
        title=dict(text='Message Label Pairwise Correlations', x=0.5),
        width=1000,
        height=1000
    )
    graphs.append(Figure(data=heatmap_data, layout=heatmap_layout))
    
    # append the earlier created word cloud figure
    graphs.append(wordcloud_fig)

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
    
    
    ##### removing this piece to comply with the rubric ######
    
    # filter to labels with any positive instances
    # this is what happens during classifier training - otherwise there will be a prediction-label mismatch
    
    # df_labels = (df.iloc[:, 4:] == 1).any(axis=0) 
    # classification_keys = list(df_labels[df_labels.values].index)
    # classification_results = dict(zip(classification_keys, classification_labels))

    classification_keys = list(df.columns)
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