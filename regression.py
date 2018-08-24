import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# Load in the built-in housing dataset
boston_housing = keras.datasets.boston_housing

(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# Shuffle the training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

# Use pandas library to display the first several rows of the dataset in a table
import pandas as pd

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']

df = pd.DataFrame(train_data, columns=column_names)
print(df.head())

# Test data is *not* used when calculating the mean and std.
# Normalizing data - (subtract the mean of the feature and divide by standard deviation)
# Makes training far easier.

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
# Saving copy of original test_data to display in table at the end of run.
original_test_data = test_data
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


# Seuquential model with two densely connected hidden layers.
# Output layer returns single, continuous value.

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu, 
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

# model = build_model()
# model.summary()

# Display training progress by printing a single dot for each completed epoch.
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 500

# Store training stats
# history = model.fit(train_data, train_labels, epochs=EPOCHS,
#                     validation_split=0.2, verbose=0,
#                     callbacks=[PrintDot()])

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

# Graph to display model's improvement.
def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']), 
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0,5])
  plt.show()

# plot_history(history)

model = build_model()

# The patience parameter is the amount of epochs to check for improvement.
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintDot()])

# plot_history(history)

# Evaluate the model using the testing data and get the total loss as well as
# Mean Absolute Error - which is a common regression metric. In this case,
# it represents the average difference in prediction price to actual price.
[loss, mae] = model.evaluate(test_data, test_labels, verbose=0)

print("\nTesting set Mean Abs Error: ${:7.2f}".format(mae * 1000))

test_predictions = model.predict(test_data).flatten()

df = pd.DataFrame(original_test_data, columns=column_names)

priceList = []
for i in range(len(df.index)):
    priceList.append(test_predictions.item(i))

df['ESTPRICE'] = priceList

print(df.head(10))
