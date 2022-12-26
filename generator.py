#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

MAJORITY OF ALL CODE SOURCED OR MODIFIED FROM DigitalSreeni
https://github.com/bnsreenu/python_for_microscopists


Load a batch of images into a numpy array
Reinterpreted and taken heavily from DigitalSreeni
https://github.com/bnsreenu/python_for_microscopists


@author: ryan
"""

# for access to os directories
import os

# our representation for our images
import numpy as np


# performs logic found in prepare_data
# simply loads a set of NumPy arrays
def load_img(directory, volumes):
    loaded = []
    
    #load each image into array
    for i, image_name in enumerate(volumes):
        if image_name.split('.')[1] == 'npy':
            #for our purposes we'll assume each image is definetly npy
            img = np.load(directory + image_name)
            print(directory + image_name)
            loaded.append(img)

    loaded = np.array(loaded) # convert to numpy array from python list
    return (loaded) # tuple for immutability




# returns a volume and the matching ground truth segmentation for use
# in the training of our model
# yields so the modal can asynchronously gets new images until it runs out
def load_batch(directory, volumes, segmentation_directory, segmentations, batch_size):
    
    l = len(volumes)
    
    # keras requires a generator execute infinitely
    # it will yield values periodically (i.e: a volume and its segmentation for training)
    
    while True:
        
        start = 0
        end = batch_size
        
        # while our batch is within the volumes available
        while start < l:
            
            # we'll stop at batch end of the last image in list
            limit = min(end, l)
            
            # load a volume and segmentation npy file!
            volume = load_img(directory, volumes[start:limit])
            segmentation = load_img(segmentation_directory, segmentations[start:limit])
            volume = volume.astype(np.float32)
            segmentation = segmentation.astype(np.float32)
            
            yield (volume, segmentation) # yield a tuple (batch for training)
            
            # increment to the next batch of training data
            start += batch_size 
            end += batch_size
            
            
# test generator ##############################################
from matplotlib import pyplot as plot
import random

volume_dir = '/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/training/volumes/'
segmentation_dir = '/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/training/segmentations/'

volume_list = sorted(os.listdir(volume_dir)) # make sure patches align
segmentation_list = sorted(os.listdir(segmentation_dir))


batch_size = 3
gen = load_batch(volume_dir, 
                         volume_list, 
                         segmentation_dir,
                         segmentation_list,
                         batch_size)

# test generator
vol, msk = gen.__next__()


num = random.randint(0, vol.shape[0]-1)
img = vol[num]
segmentation = msk[num]
segmentation = np.argmax(segmentation, axis=3)

n = random.randint(0, segmentation.shape[2])
plot.figure(figsize=(16,12))

plot.subplot(221)
plot.imshow(img[:,:,n, 0], cmap='gray')
plot.title('Flair')

# plot T1CE
plot.subplot(222)
plot.imshow(img[:,:,n, 1], cmap='gray')
plot.title('T1CE')

# plot T2
plot.subplot(223)
plot.imshow(img[:,:,n, 2], cmap='gray')
plot.title('T2')

# plot segmentation
plot.subplot(224)
plot.imshow(segmentation[:,:,n])
plot.title('Segmentation segmentation')
############################################################

