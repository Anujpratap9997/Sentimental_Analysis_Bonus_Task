from flask import Flask, jsonify, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

app = Flask(__name__)

# Load the dataset
df = pd.read_csv('IMDB Dataset.csv')

# Preprocess the data
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['review'].values)

X = tokenizer.texts_to_sequences(df['review'].values)
X = pad_sequences(X, maxlen=200)

y = df['sentiment'].values

# Train Word2Vec model
sentences = [review.split() for review in df['review'].values]
word2vec_model = Word2Vec(sentences=sentences, size=100, window=5, min_count=1, workers=4)

# Convert text data to word embeddings
embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 100))
for word, i in tokenizer.word_index.items():
    if word in word2vec_model.wv:
        embedding_matrix[i] = word2vec_model.wv[word]

# Build the RNN model
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, weights=[embedding_matrix], input_length=200, trainable=False))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, batch_size=128, epochs=10, validation_split=0.2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    input_data = tokenizer.texts_to_sequences([text])
    input_data = pad_sequences(input_data, maxlen=200)
    prediction = model.predict(input_data)[0][0]
    if prediction >= 0.5:
        output = 'Positive'
    else:
        output = 'Negative'
    return render_template('index.html', prediction_text='Sentiment: {}'.format(output))

if __name__ == '__main__':
    app.run(debug=True)
