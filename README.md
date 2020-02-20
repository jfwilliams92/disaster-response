# Disaster Response Pipeline Project

## Background and Motivation
During times of disaster, disaster response teams are faced with the daunting task of effectively orchestrating aid efforts to those who are in need. Those in need may attempt to contact those heading aid operations to secure aid for themselves. Other times they may be unable to communicate directly with the aid operatives and instead resort to asking for assistance via social media. <br>
However, separating the messages/requests of those in need from the deluge of communications and social media posts that are generated during times of disaster can be likened to finding a needle in a haystack. To assist rescue operatives in identifying communications representing individuals in true need, we will be deploying a web-based machine learning model capable of differentiating between signal (individual(s) in need) and noise (non-related communications and posts). The model will assign labels to messages out of a pool of potential labels.
<p>
Additionally, we'll provide data cleansing and model training scripts to allow new model(s) to be trained and deployed from the existing dataset or a new dataset.
<p>
The dataset for our model building comes from Figure Eight, and is comprised of thousands of messages generated during times of disaster, including both messages sent directly to aid originizations and messages posted to social media.
<p>
Within this repository, I have provided two Jupyter notebooks that outline the process involved in cleaning the data and training a model to predict labels from text features. Feel free to explore disaster_response_app/exploration.

## Purpose and Functionality of the Web Application
We will be deploying a web application to allow disaster responders to classify incoming messages. The web application will accept text, clean and extract features from the text, and generate a prediction for each potential message label from our trained model.<br> If you select 'Hard Label Classification', only the labels for which the model predicted positively will appear.<br> If you select 'Probability of Label', all potential labels will appear, color-coded according to the model's predicted probability of the label being true. 

* Labels with a predicted probability <= 33% will appear red. 
* Labels with a predicted probability between 34% and 67% will appear yellow.
* Labels with a predicted probability > 67% will appear green.

## Installation and Setup Instructions
In order to set up this application to run locally on your machine, clone this repository or download as a .zip file. <br>
Once you have downloaded the repository, navigate to it. <br>
At this point I would recommend that you create a virtual environment to isolate the packages required to run this app. If you are using Anaconda: <br>
`> conda create -n new_env_name python=3.7`<br>
`> conda source activate new_env_name`<br>
Then, you can install all the packages required for this application with pip: <br>
`> pip install -r requirements.txt` 
<p>

Now you are ready to setup and run the application. Navigate into the disaster_response_app folder: <br>
`> cd disaster_response_app` <br>
You will need to run several setup scripts before the application is ready to be started. <br> First, load and clean the message and categories data, and save the cleaned data off into a local sqlite3 database file. <br> Provide the location of the messages csv, the location of the categories csv, and the location to which you wish to save your database file. <br> Example: <br>
`> python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterMessages.db`
<br>
Then, train and save a model from the cleaned data. <br> Provide the location of the cleaned messages database file and the path to which you wish to save your trained model.<br> 
Example: <br>
`> python models/train_classifier.py data/DisasterMessages.db models/classifier.pkl`<br>
Finally, create the data necessary for the word cloud visualization. <br>
`> python models/word_cloud.py`

## Running the web app
In order to run the web application, run the following command from within disaster_response_app: <br>
`> python run.py` <br>
After a few seconds the web application will start up. In your terminal you will see the URL to navigate to your locally running application. <br>
Feel free to explore the visualizations and classify some messages!
