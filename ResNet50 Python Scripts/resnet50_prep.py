import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import scipy
from PIL import Image
from scipy import ndimage

import os
import h5py

'''Number of pixels per side of each resized, square training image.'''
# adjust this, if necessary (keep a power of 2)
num_px = 128

X_train = []
Y_train = []

breed_num = 0

'''Iterate through breeds.'''
with open('../Classes/breeds.txt') as file:
    for line in file:
        # Extract breed
        breed = line.strip()
        print(breed, breed_num)

        for file in os.listdir('../Stanford_Dogs_Dataset/' + breed + '/'):
            try:
                img_path = '../Stanford_Dogs_Dataset/' + breed + '/' + file
                img = image.load_img(img_path, target_size=(num_px, num_px))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                # print('Input image shape:', x.shape)

                X_train.append(x)
                Y_train.append(breed_num)
            except:
                continue

        print(len(X_train))
        breed_num += 1

# Concatenate and save image vectors
X_train = np.concatenate(X_train, axis = 0)
print(X_train.shape)

with h5py.File("train.h5", "w") as f:
    f.create_dataset('data_X', data=X_train)
    f.create_dataset('data_Y', data=Y_train)

