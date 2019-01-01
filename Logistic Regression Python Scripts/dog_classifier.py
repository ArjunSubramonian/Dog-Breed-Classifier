# Arjun Subramonian

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage

import operator

# Helper functions

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def predict(w, b, X):
    # Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    # Arguments:
    # w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    # b -- bias, a scalar
    # X -- data of size (num_px * num_px * 3, number of examples)
    
    # Returns:
    # A -- a numpy array (vector) containing all probabilities of a breed being present in the picture
    
    w = w.reshape(X.shape[0], 1)
    
    # Compute A, which predicts the probabilities of a breed being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)
    
    return A

print()
print()
print()

# Change fname to test image name.
fname = "../image.jpg"
num_px = 128

# Reshape image into single column vector.
image = np.array(ndimage.imread(fname, flatten = False))
### CAN INCORPORATE ASPECT SCALING AND CROPPING FUNCTIONS HERE!!!
my_image = image.reshape((1, num_px * num_px * 3)).T

'''Make a dictionary with the breeds as keys.'''
breeds = {}
with open('../Classes/breeds.txt') as file:
    for line in file:
        breed = line.strip()

        w = np.loadtxt('../trained_values/' + breed + '_w.txt')
        b = np.loadtxt('../trained_values/' + breed + '_b.txt')
        mini = np.loadtxt('../trained_values/' + breed + '_mini.txt')
        maxi = np.loadtxt('../trained_values/' + breed + '_maxi.txt')

        # Scale RGB values to [0, 1] using mini and maxi.
        img = (my_image - mini) / (maxi - mini)

        prediction = np.squeeze(predict(w, b, img))

        breeds[breed] = prediction * 100

        # Print breed if P(breed = 1 | input image) > 95%
        if prediction * 100 > 95:
            print(breed, str(prediction * 100) + '%')

print()
print()
print()

# Print top 5 P(breed = 1 | input image)'s
n = 5
top_n = sorted(breeds.items(), key = operator.itemgetter(1), reverse = True)[:n]
for breed in top_n:
    print(breed[0], str(breed[1]) + '%')

print()
print()
print()
