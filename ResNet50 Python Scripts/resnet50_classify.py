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

'''Number of pixels per side of each resized, square training image.'''
# adjust this, if necessary (keep a power of 2)
num_px = 128

breeds = {}
breed_num = 0

'''Iterate through breeds.'''
with open('../Classes/breeds.txt') as file:
    for line in file:
        breeds[breed_num] = line.strip()
        breed_num += 1

img_path = '../image.jpg'
img = image.load_img(img_path, target_size=(num_px, num_px))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

model = load_model('my_model.h5')

print()
print()
print()

ind = np.argmax(model.predict(x))
print('Breed:', breeds[ind])

print()
print()
print()
