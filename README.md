# Extended-tensorflow-tutorials
Extension of tutorials found here: https://www.tensorflow.org/tutorials/keras/basic_classification
and here: https://www.tensorflow.org/tutorials/keras/basic_text_classification

# Simple Classification
## Steps to run program
### Install tensorflow
You'll need to install tensorflow on whatever system you are using here: https://www.tensorflow.org/install/
### Personal setup
I used Anaconda to set up the tensorflow environment. 
You can install anaconda here: https://www.anaconda.com/download/
### To run
1. Confirm you have a working tensorflow environment (following instructions on website). 
2. Then, you simply need to run in your console:
```
python simple-classification.py
```
3. You should see the dataset being loaded, then a separate window should pop up showing a collection of the images
being processed.

4. After you close the pop-up image, the model will initialize training through 5 epochs by default.

5. Another window should pop up displaying images and whether or not they were classified correctly by the model.

6. After closing that window, you will be prompted to enter integers. These integers are used to determine indices in the test
dataset. Once a correct integer is input, the model predicts its classification and it is printed to the console.

7. A separate window will pop up to display the selected image and if it was classified correctly.

8. After that window is closed, the program will continue to allow you to enter inputs until you provide an integer outside
the bounds of the numpy array. 

# Text Classification
## Steps to run program
### Install tensorflow
See above
### Personal setup
See above
### To run
1. Confirm a working tensorflow environment.
2. 2. Then, you simply need to run in your console:
```
python text-classification.py
```
3. Imdb dataset should be loaded for the first run, and cached for future runs.
4. Information on dataset / model is printed to console.
5. Model is trained.
6. TBA

