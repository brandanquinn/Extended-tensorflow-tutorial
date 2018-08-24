import tensorflow as tf
from tensorflow import keras

import numpy as np

import string

print(tf.__version__)

# Loading built-in IMDB review dataset
# To use a different dataset of text, must be sliced in a similar format here.
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()} 
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# In order to process the reviews through the model, they must be encoded to unique integers. 
# Since the dataset is handled by tensorflow internally - it starts as the encoded list.
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# input shape is the vocabulary count used for the movie reviews (10,000 words)
vocab_size = 10000

# set up model
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

# print model information to console
model.summary()

# Loss function and optimizer set for training purposes. Since we are classifying in
# a binary manor (positive/negative) we use the binary_crossentropy function for loss.
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Slicing the dataset to get training and testing values.
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# Train the model based on the sliced training values. Ran over 40 epochs with batch size of 512.
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# Evaluate the trained model with testing data and compute result.
results = model.evaluate(test_data, test_labels)
print(results)

# Live coding done below:
