#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 19:59:29 2022

Using BraTS data for prediction and comparison to ground truth

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
crop = lambda volume=None : volume[56:184, 56:184, 13:141] # 128x128x128 - we need a cube for unet 


# globals 
validation_path = '/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation2/'
output_path = '/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/'

# get the different images in the subdirectory of the validation path
t1ce_paths = sorted(glob.glob(validation_path + '*/*t1ce.nii')) # glob gets image by pathname that matches a pattern
t2_paths = sorted(glob.glob(validation_path + '*/*t2.nii'))
flair_paths = sorted(glob.glob(validation_path + '*/*flair.nii'))
segmentation_paths = sorted(glob.glob(validation_path + '*/*seg.nii'))


# create a combined volume and segmented volume from each one
for vol_num in range(0, len(segmentation_paths)):
     
    print('Processing validation volume', vol_num)
    
    validation_volumes = []
    validation_segmentations = []

    # load current volumes images
    t1ce = nb.loadsave.load(t1ce_paths[vol_num]).get_fdata()
    t2 = nb.loadsave.load(t2_paths[vol_num]).get_fdata()
    flair = nb.loadsave.load(flair_paths[vol_num]).get_fdata()
    segmentation = nb.loadsave.load(segmentation_paths[vol_num]).get_fdata()
    
    # for ease of comparison, scale the validation images
    t1ce = scaleNiFTi(t1ce)
    t2 = scaleNiFTi(t2)
    flair = scaleNiFTi(flair)
    
    # for use with the model, images need to be a power of 2 cube
    t1ce = crop(t1ce)
    t2 = crop(t2)
    flair = crop(flair)
    
    # now combine the different modals into 1 multi modal array
    combined = np.stack([flair, t1ce, t2], axis=3)
    
    segmentation = segmentation.astype(np.uint8)
    
    # replace 4 with 3 in the array since 3 is unused
    # will play an important role in create 3D UNet
    segmentation[segmentation == 4] = 3
    
    # crop combined multi-modal, and segmentation
    segmentation = crop(segmentation)
    segmentation = to_categorical(segmentation, num_classes=4)
    
    validation_volumes.append(combined)
    validation_segmentations.append(segmentation)

    # convert to numpy array
    validation_volumes = np.array(validation_volumes)
    validation_segmentations = np.array(validation_segmentations)
    
    for i, vol in enumerate(validation_volumes):
        np.save(output_path + 'volumes/volume_' + str(vol_num) + '_patch_' + str(i) +'.npy', vol)
        np.save(output_path + 'segmentations/segmentation_' + str(vol_num) + '_patch_' + str(i) +'.npy', validation_segmentations[i])


# pick a volume for prediction and comparison ##################################
# pick one volume
import random
vol_num = random.randint(0, len(validation_segmentations))
print("For validation image:", vol_num)

vol = validation_volumes[vol_num]
seg = validation_segmentations[vol_num]

# pick a random slice out of all in the segmentation
slice_num = random.randint(0, len(seg[2]) - 1)
print("Slice number:", slice_num)

plot.figure(figsize=(12, 8)) # keyword argument for tuple

# plot flair
plot.subplot(231)
plot.imshow(vol[:,:,slice_num, 0], cmap='gray')
plot.title('Flair')

# plot T1CE
plot.subplot(232)
plot.imshow(vol[:,:,slice_num, 1], cmap='gray')
plot.title('T1CE')

# plot T2
plot.subplot(233)
plot.imshow(vol[:,:,slice_num, 2], cmap='gray')
plot.title('T2')

# plot segmentation
plot.subplot(235)
plot.imshow(seg[:,:,slice_num, 2])
plot.title('Segmentation mask')

plot.show()
################################################################################

# load the model in for prediction #############################################
import keras
from keras.models import load_model
my_model = load_model('brats_3d.hdf5', 
                      compile=False) # no need to compile since we're only predicting





expanded = np.expand_dims(vol, axis=0)
prediction_test = my_model.predict(expanded)
prediction_test_argmax=np.argmax(prediction_test, axis=4)[0,:,:,:] #take the highest modality


plot.figure(figsize=(12, 8))
plot.subplot(231)
plot.title('Testing Image')
plot.imshow(vol[:,:,slice_num,2], cmap='gray')
plot.subplot(232)
plot.title('Ground-Truth Image')
plot.imshow(seg[:,:,slice_num,2])
plot.subplot(233)
plot.title('Prediction on test image')
plot.imshow(prediction_test_argmax[:,:, slice_num])

difference = np.subtract(prediction_test_max, seg[:,:,:, 3])

plot.subplot(234)
plot.title('Difference on test image')
plot.imshow(difference[:,:, slice_num])
plot.show()


################################################################################