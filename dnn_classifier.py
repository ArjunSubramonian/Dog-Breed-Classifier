# Arjun Subramonian

import numpy as np
import scipy
from PIL import Image
from scipy import ndimage
from dnn_helper import *

import operator

# Helper function

def predict(X, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    
    # Forward propagation
    p, caches = L_model_forward(X, parameters)
        
    return p

print()
print()
print()

def preprocess(fname):
    num_px = 128

    try:
        img = Image.open(fname)

        # resize: (width, height)
        # crop: (left, upper, right, lower)
        if img.size[0] < img.size[1]:
            basewidth = num_px
            wpercent = (basewidth / float(img.size[0]))
            hsize = int((float(img.size[1]) * float(wpercent)))
            img = img.resize((basewidth, hsize), Image.ANTIALIAS)

            width, height = img.size
            new_width, new_height = num_px, num_px

            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            img = img.crop((left, top, right, bottom))

            img.save(fname)
        else:
            sideheight = num_px
            hpercent = (sideheight / float(img.size[1]))
            wsize = int((float(img.size[0]) * float(hpercent)))
            img = img.resize((wsize, sideheight), Image.ANTIALIAS)

            width, height = img.size
            new_width, new_height = num_px, num_px

            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2

            img = img.crop((left, top, right, bottom))

            img.save(fname)

        return 1
    except:
        return -1

# Change fname to test image name.
fname = "image.jpg"
preprocess(fname)

num_px = 128
L = 4

# Reshape image into single column vector.
image = np.array(ndimage.imread(fname, flatten = False))
my_image = image.reshape((1, num_px * num_px * 3)).T

'''Make a dictionary with the breeds as keys.'''
breeds = {}
with open('test.txt') as file:
    for line in file:
        breed = line.strip()

        parameters = {}
        for l in range(1, L + 1):
            parameters["W" + str(l)] = np.loadtxt('./trained_values/' + breed + '_W' + str(l) + '.txt')
            b = np.loadtxt('./trained_values/' + breed + '_b' + str(l) + '.txt')
            try:
                parameters["b" + str(l)] = np.reshape(b, (b.shape[0], 1))
            except:
                parameters["b" + str(l)] = np.reshape(b, (1, 1))
        
        mini = np.loadtxt('./trained_values/' + breed + '_mini.txt')
        maxi = np.loadtxt('./trained_values/' + breed + '_maxi.txt')

        # Scale RGB values to [0, 1] using mini and maxi.
        img = (my_image - mini) / (maxi - mini)

        pred = np.squeeze(predict(img, parameters))

        breeds[breed] = pred * 100

        # Print breed if P(breed = 1 | input image) > 95%
        # if pred * 100 > 95:
        print(breed, str(pred * 100) + '%')

print()
print()
print()

# Print top 5 P(breed = 1 | input image)'s
# n = 5
# top_n = sorted(breeds.items(), key = operator.itemgetter(1), reverse = True)[:n]
# for breed in top_n:
    # print(breed[0], str(breed[1]) + '%')

# print()
# print()
# print()
