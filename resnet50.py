# References the ResNet algorithm due to He et al. (2015), and takes significant inspiration and follows the structure given in the github repository of Francois Chollet:  
    # - Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun - [Deep Residual Learning for Image Recognition (2015)](https://arxiv.org/abs/1512.03385)
    # - Francois Chollet's github repository: https://github.com/fchollet/deep-learning-models/blob/master/resnet50.py
# Uses Keras.
# Uses weights pretrained on ImageNet.

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
from keras.callbacks import ModelCheckpoint

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

import os
import h5py

'''Number of pixels per side of each resized, square training image.'''
# adjust this, if necessary (keep a power of 2)
num_px = 128
num_breeds = 120

# In ResNets, a "shortcut" or a "skip connection" allows the gradient to be directly backpropagated to earlier layers:  

# Two main types of blocks are used in a ResNet, depending mainly on whether the input/output dimensions are same or different.

# The identity block is the standard block used in ResNets, and corresponds to the case where the input activation has the same dimension as the output activation. 
    # The upper path is the "shortcut path." The lower path is the "main path." To speed up training we have also added a BatchNorm step. 
    # Implements a slightly more powerful version of the identity block, in which the skip connection "skips over" 3 hidden layers rather than 2 layers.

    # First component of main path: 
    # - The first CONV2D has F1 filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be conv_name_base + '2a'. Uses 0 as the seed for the random initialization. 
    # - The first BatchNorm is normalizing the channels axis.  Its name should be bn_name_base + '2a'.
    # - Applies the ReLU activation function. 
    
    # Second component of main path:
    # - The second CONV2D has F2 filters of shape (f,f) and a stride of (1,1). Its padding is "same" and its name should be conv_name_base + '2b'. Uses 0 as the seed for the random initialization. 
    # - The second BatchNorm is normalizing the channels axis.  Its name should be bn_name_base + '2b'.
    # - Applies the ReLU activation function.
    
    # Third component of main path:
    # - The third CONV2D has F3 filters of shape (1,1) and a stride of (1,1). Its padding is "valid" and its name should be conv_name_base + '2c'. Uses 0 as the seed for the random initialization. 
    # - The third BatchNorm is normalizing the channels axis.  Its name should be bn_name_base + '2c'. There is no ReLU activation function in this component. 
    
    # Final step: 
    # - The shortcut and the input are added together.
    # - Applies the ReLU activation function.

def identity_block(X, f, filters, stage, block):
    """
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. Add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Adds shortcut value to main path, and passes it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

# The ResNet "convolutional block" is the other type of block. Use this type of block when the input and output dimensions don't match up. The difference with the identity block is that there is a CONV2D layer in the shortcut path. 
# The CONV2D layer in the shortcut path is used to resize the input to a different dimension, so that the dimensions match up in the final addition needed to add the shortcut value back to the main path.
    # First component of main path:
    # - The first CONV2D has F1 filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be conv_name_base + '2a'. 
    # - The first BatchNorm is normalizing the channels axis.  Its name should be bn_name_base + '2a'.
    # - Applies the ReLU activation function.
    
    # Second component of main path:
    # - The second CONV2D has F2 filters of (f,f) and a stride of (1,1). Its padding is "same" and its name should be conv_name_base + '2b'.
    # - The second BatchNorm is normalizing the channels axis.  Its name should be bn_name_base + '2b'.
    # - Applies the ReLU activation function. 
    
    # Third component of main path:
    # - The third CONV2D has F3 filters of (1,1) and a stride of (1,1). Its padding is "valid" and its name should be conv_name_base + '2c'.
    # - The third BatchNorm is normalizing the channels axis.  Its name should be bn_name_base + '2c'. There is no ReLU activation function in this component. 

    # Shortcut path:
    # - The CONV2D has F3 filters of shape (1,1) and a stride of (s,s). Its padding is "valid" and its name should be conv_name_base + '1'.
    # - The BatchNorm is normalizing the channels axis.  Its name should be bn_name_base + '1'. 
    
    # Final step: 
    # - The shortcut and the main path values are added together.
    # - Applies the ReLU activation function. 

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used
    
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First component of main path 
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', name = conv_name_base + '1', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Adds shortcut value to main path, and passes it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

# ResNet model (50 layers)
    # - Zero-padding pads the input with a pad of (3,3)
    # - Stage 1:
    #     - The 2D Convolution has 64 filters of shape (7,7) and uses a stride of (2,2). Its name is "conv1".
    #     - BatchNorm is applied to the channels axis of the input.
    #     - MaxPooling uses a (3,3) window and a (2,2) stride.
    # - Stage 2:
    #     - The convolutional block uses three set of filters of size [64,64,256], "f" is 3, "s" is 1 and the block is "a".
    #     - The 2 identity blocks use three set of filters of size [64,64,256], "f" is 3 and the blocks are "b" and "c".
    # - Stage 3:
    #     - The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
    #     - The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
    # - Stage 4:
    #     - The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
    #     - The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
    # - Stage 5:
    #     - The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
    #     - The 2 identity blocks use three set of filters of size [512, 512, 2048], "f" is 3 and the blocks are "b" and "c".
    # - The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
    # - The Fully Connected (Dense) layer reduces its input to the number of classes using a softmax activation. Its name should be 'fc' + str(classes).


### EDIT ###
def ResNet50(input_shape = (num_px, num_px, 3), classes = num_breeds):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=2, block='c')

    # Stage 3
    X = convolutional_block(X, f = 3, filters = [128,128,512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [128,128,512], stage=3, block='b')
    X = identity_block(X, 3, [128,128,512], stage=3, block='c')
    X = identity_block(X, 3, [128,128,512], stage=3, block='d')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
    X = identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), name = 'avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    # WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
    # weights_path = get_file('resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', WEIGHTS_PATH_NO_TOP, cache_subdir='models', md5_hash='a268eb855778b3df3c7506639542a6af')
    # model.load_weights(weights_path, by_name=True)
    model.load_weights('weights-improvement-01-0.00.h5')

    return model

### TRAIN THE MODEL ###
model = ResNet50(input_shape = (num_px, num_px, 3), classes = num_breeds)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_dataset = h5py.File('train.h5', "r")
train_set_x_orig = np.array(train_dataset["data_X"][:]) # your train set features
train_set_y_orig = np.array(train_dataset["data_Y"][:]) # your train set labels
train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))

# Normalize
X_train = train_set_x_orig / 255

# Convert training labels to one hot matrices
Y_train = convert_to_one_hot(train_set_y_orig, num_breeds).T

# Checkpoint model
filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
model.fit(X_train, Y_train, epochs = 50, batch_size = 64, callbacks=callbacks_list, validation_split=0.1)

# Save model
model.save('my_model.h5')
