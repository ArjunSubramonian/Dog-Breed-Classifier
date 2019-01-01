# Arjun Subramonian

import numpy as np
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

import os

'''Make a dictionary with the breeds as keys.'''
breeds = {}
with open('../Classes/breeds.txt') as file:
    for line in file:
        breeds[line.strip()] = []

'''Number of pixels per side of each resized, square training image.'''
# adjust this, if necessary (keep a power of 2)
num_px = 128

'''Iterate through breeds.'''
for breed in breeds:
    print(breed)

    # Number of training examples
    num_pos = len([name for name in os.listdir('../Stanford_Dogs_Dataset/' + breed + '/')])

    pos_X = np.zeros((num_px * num_px * 3, num_pos))
    pos_Y = np.ones((1, num_pos))

    col = 0
    for file in os.listdir('../Stanford_Dogs_Dataset/' + breed + '/'):
        try:
            image = np.array(ndimage.imread('../Stanford_Dogs_Dataset/' + breed + '/' + file, flatten = False))
            # Reshape image into single column vector.
            my_image = image.reshape((1, num_px * num_px * 3)).T

            pos_X[:, col] = my_image[:, 0]

            col += 1
        except:
            continue

    # Check matrix sizes of positive training data and labels.
    assert(pos_X.shape == (num_px * num_px * 3, num_pos))
    assert(pos_Y.shape == (1, num_pos))

    np.savetxt('../data/' + breed + '_X.txt', pos_X.astype(int), fmt = '%i')
    np.savetxt('../data/' + breed + '_Y.txt', pos_Y.astype(int), fmt = '%i')
