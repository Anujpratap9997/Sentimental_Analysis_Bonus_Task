# Sentimental_Analysis_Bonus_Task
Overview
This repository contains code for a deep learning model that predicts the sentiment (positive or negative) of a movie review. The model is built using a Recurrent Neural Network (RNN) and Word2Vec embeddings.

There are mainly two code files in this repository:

model.py: contains code for the RNN model
api.py: contains code for a Flask-based RESTful API that serves the model
Model
The RNN model is built using TensorFlow and Keras. It has an embedding layer followed by an LSTM layer and a dense output layer. The input to the model is a sequence of integers representing the words in a movie review. These integer sequences are created using the Tokenizer class from Keras.

The model is trained on the IMDB Movie Review Dataset and achieves an accuracy of around 87% on the test set.

API
The API is built using Flask, a Python web framework. The API accepts POST requests containing movie reviews as text and returns the sentiment of the review as a string ('Positive' or 'Negative').

To use the API, you need to have the requests Python package installed. 

Files
model.py: contains the code for the RNN model
api.py: contains the code for the Flask-based RESTful API that serves the model
sentiment_model.h5: contains the trained RNN model weights
templates/: a directory containing the HTML templates for the API's web pages
IDMV_dataset.csv: Contain the dataset of IDMV movies reviews
Requirements
Python 3.x
TensorFlow 2.x
Flask
Numpy
Pandas
Gensim
These packages can be installed using pip.




