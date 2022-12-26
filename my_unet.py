#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

MAJORITY OF ALL CODE SOURCED OR MODIFIED FROM DigitalSreeni
https://github.com/bnsreenu/python_for_microscopists


Utilize data preparation and generator
Adapted from DigitalSreeni

OUR MODEL ARCHITECTURE
LACKS A PAIR OF CONTRACTION AND EXPANSION OPERATIONS TO DECREASE TRAINABLE PARAMETERS
LACKS ONE ORIGINAL DOWNSAMPLING TO MAINTAIN SPATIAL INFORMATION ON 64X64X64 IMAGES

@author: ryan
"""

from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, concatenate, Conv3DTranspose, BatchNormalization, Dropout, Lambda
from keras.optimizers import Adam
from keras.metrics import MeanIoU

kernel_initializer =  'he_uniform'


def simple_unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS, num_classes):
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH, IMG_CHANNELS))
    s = inputs

    #Contraction path
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(s)
    conv1 = Dropout(0.1)(conv1)
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(conv1)
    pool1 = MaxPooling3D((2, 2, 2))(conv1)
    
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(pool1)
    conv2 = Dropout(0.1)(conv2)
    conv2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(conv2)
    pool2 = MaxPooling3D((2, 2, 2))(conv2)
     
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(conv3)
    pool3 = MaxPooling3D((2, 2, 2))(conv3)
     
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(pool3)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv3D(128, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    # Expansion operations
    upsample1 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv4)
    upsample1 = concatenate([upsample1, conv3])
    unConv1 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(upsample1)
    unConv1 = Dropout(0.2)(unConv1)
    unConv1 = Conv3D(64, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(unConv1)
     
    upsample2 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(unConv1)
    upsample2 = concatenate([upsample2, conv2])
    unConv2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(upsample2)
    unConv2 = Dropout(0.1)(unConv2)
    unConv2 = Conv3D(32, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(unConv2)
     
    upsample3 = Conv3DTranspose(16, (2, 2, 2), strides=(2, 2, 2), padding='same')(unConv2)
    upsample3 = concatenate([upsample3, conv1])
    unConv3 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(upsample3)
    unConv3 = Dropout(0.1)(unConv3)
    unConv3 = Conv3D(16, (3, 3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(unConv3)
     
    outputs = Conv3D(num_classes, (1, 1, 1), activation='softmax')(unConv3)
     
    model = Model(inputs=[inputs], outputs=[outputs])
    #compile model outside of this function to make it flexible. 
    model.summary()
    
    return model

# test on our multiple input types
model = simple_unet_model(64, 64, 64, 3, 4)
print(model.input_shape)
print(model.output_shape)

# test on our multiple input types
model = simple_unet_model(128, 128, 128, 3, 4)
print(model.input_shape)
print(model.output_shape)
