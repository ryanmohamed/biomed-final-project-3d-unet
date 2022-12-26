#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 22:20:38 2022

Testing grounds for necessary data quality assurance
Some notes are that we must
1. Scale the different modals
2. Crop unnecessary blank information
3. Combine channels 

Reinterpreted and taken heavily from DigitalSreeni
https://github.com/bnsreenu/python_for_microscopists

@author: ryan
"""

# used to process NiFTi files
import nibabel as nb
from nibabel.testing import data_path

# used to alleviate process of saving images
# rather we can save as numpy array with information
# from all modals
import numpy as np
import glob


# convert volumes to categorical for multi modal segmentation
from tensorflow.keras.utils import to_categorical

# for visualization
import matplotlib.pyplot as plot


# rmr because all the images are in 16-bit
# their intensity ranges fluncuate
# to make sure they're mapped to the same range
# we scale the images

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # scalar obj


# helper lambda functions
scaleNiFTi = lambda img=None : scaler.fit_transform( img.reshape(-1, img.shape[1]) ).reshape(img.shape)


# initial tests ########################################
training_path = '/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/training/'

# load an image from training path using nibabel
test_flair = nb.loadsave.load(training_path + 'BraTS_Training_001/BraTS20_Training_001_flair.nii').get_fdata()
print(test_flair.max()) # numpy array

# notice the peak value in this image is 625.0
# load another for comparison
test_flair2 = nb.loadsave.load(training_path + 'BraTS_Training_005/BraTS20_Training_005_flair.nii').get_fdata()
print(test_flair2.max()) # numpy array

# peak value in this case is 762! we want to scale these down


# bc scalars from MinMaxScaler are applied to 1D arrays
# flatten the array and reshape!
test_flair = scaleNiFTi(test_flair)
print(test_flair.max()) # max value is now 1 

# now scale the rest of the modals
test_t1 = nb.loadsave.load(training_path + 'BraTS_Training_001/BraTS20_Training_001_t1.nii').get_fdata()
test_t1ce = nb.loadsave.load(training_path + 'BraTS_Training_001/BraTS20_Training_001_t1ce.nii').get_fdata()
test_t2 = nb.loadsave.load(training_path + 'BraTS_Training_001/BraTS20_Training_001_t2.nii').get_fdata()
test_mask = nb.loadsave.load(training_path + 'BraTS_Training_001/BraTS20_Training_001_seg.nii').get_fdata()

# scale the other modals
test_t1 = scaleNiFTi(test_t1)
test_t1ce = scaleNiFTi(test_t1ce)
test_t2 = scaleNiFTi(test_t1ce)

# now for the mask we obviously dont have to scale
# but let's check it out
print(np.unique(test_mask))

# notice how we use float point number which requires more space
# convert to integer type, we're working with labels!
test_mask = test_mask.astype(np.uint8)
print(np.unique(test_mask))

# replace 4 with 3 now
test_mask[ test_mask == 4 ] = 3
print(np.unique(test_mask))

# visualize a single slice
import random
n = random.randint(0, test_mask.shape[2])

plot.figure(figsize=(12, 8)) # keyword argument for tuple

plot.subplot(231)
plot.imshow(test_flair[:,:,n], cmap='gray')
plot.title('Flair')

plot.subplot(232)
plot.imshow(test_t1[:,:,n], cmap='gray')
plot.title('T1')

plot.subplot(233)
plot.imshow(test_t1ce[:,:,n], cmap='gray')
plot.title('T1CE')

plot.subplot(234)
plot.imshow(test_t2[:,:,n], cmap='gray')
plot.title('T2')

plot.subplot(235)
plot.imshow(test_mask[:,:,n])
plot.title('Segmentation mask')

plot.show()

# combine the images into a single volume used for training later
# ONLY channels with information
combined_x = np.stack([test_flair, test_t1ce, test_t2], axis=3) # produces 4d array, 4th represent channel

# as you can seen in the original data, unnecessary blank information
# our network can become biased towards images with blank information
# which is a negative, so crop the image to 128x128x128

# 128 is too small and omits parts of the volume! 
# modified to 140x175x170 (common dimensions found)

combined_x = combined_x[49:189, 43:218, 0:140]


# modificaton, because my computer is weak, i'll implement
# a patch based method for training
# patches will be 28x35x28
# 125 patches from one image
# field may be too small! we'll see

patches = []
for hs in np.hsplit(combined_x, 5):
    
    for vs in np.vsplit(hs, 5):
        
        for ds in np.dsplit(vs, 5):
            patches.append(ds)
            
patches = np.array(patches)

patch = patches[random.randint(0, len(patches) - 1)]
n = random.randint(0, patch.shape[2])
plot.figure(figsize=(12, 8)) # keyword argument for tuple


# plot flair
plot.subplot(231)
plot.imshow(patch[:,:,n, 0], cmap='gray')
plot.title('Flair')

# plot T1CE
plot.subplot(232)
plot.imshow(patch[:,:,n, 1], cmap='gray')
plot.title('T1CE')

# plot T2
plot.subplot(233)
plot.imshow(patch[:,:,n, 2], cmap='gray')
plot.title('T2')

# plot segmentation
plot.subplot(235)
plot.imshow(patch[:,:,n])
plot.title('Segmentation mask')

plot.show()


# n = random.randint(0, test_mask.shape[2])

plot.figure(figsize=(12, 8)) # keyword argument for tuple


# plot flair
plot.subplot(231)
plot.imshow(combined_x[:,:,n, 0], cmap='gray')
plot.title('Flair')

# plot T1CE
plot.subplot(232)
plot.imshow(combined_x[:,:,n, 1], cmap='gray')
plot.title('T1CE')

# plot T2
plot.subplot(233)
plot.imshow(combined_x[:,:,n, 2], cmap='gray')
plot.title('T2')

# plot segmentation
plot.subplot(235)
plot.imshow(test_mask[:,:,n])
plot.title('Segmentation mask')

plot.show()


# save as numpy to avoid unnecessary image save with multichannels
np.save(training_path + '/combined001', combined_x)

# double check we can reinterpret with it
test_open = np.load(training_path + '/combined001.npy')

# convert to categorical multi modal image
test_mask = to_categorical(test_mask, num_classes=4) # pretty much makes 3d - 4d for accessing specific channels

