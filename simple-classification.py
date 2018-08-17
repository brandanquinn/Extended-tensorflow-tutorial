import tensorflow as tf
from tensorflow import keras

import numpy as np
# Needed to set a mpl backend variable
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
print(tf.__version__)

# Load built-in image dataset.
# To use a different dataset of images, must be sliced in a similar format here.
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# List of classifications
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# Preprocessing of the data done in order to scale the pixel values into floats within the range of 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Visualize examples of images within the dataset.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.show()    

# Initialize the model with Dense layers and Flattens format of images from 2d (28x28) to 1d array of 784 pixels.
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Go-to optimizer, sparse_categorical_crossentropy used since we have several possible classifications.
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model.
model.fit(train_images, train_labels, epochs=5)

# Compute the accuracy of the trained model using the sliced test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)

# Gets predictions for test images to display in visualization below.
predictions = model.predict(test_images)

# Plot the first 25 test images, their predicted label, and the true label
# Color correct predictions in green, incorrect predictions in red
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
plt.show()

# continuously test input in trained model
# have user input number, run that number test_image through model
# get prediction, display image

user_input = 0

while True:
    user_input = int(input('Enter a number between 0 and 10000: '))
    if user_input < 0 or user_input >= 10000: 
        print('Input out of bounds.')
        break
    img = test_images[user_input]
    img = (np.expand_dims(img, 0))

    predictions = model.predict(img)
    print('Test image predicted value: ', np.argmax(predictions[0]))

    plt.figure()
    plt.imshow(test_images[user_input])
    predicted_label = np.argmax(predictions[0])
    true_label = test_labels[user_input]
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label]),
                                  color=color)
    plt.show()
