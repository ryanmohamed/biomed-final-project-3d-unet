# -*- coding: utf-8 -*-
"""

MAJORITY OF ALL CODE SOURCED OR MODIFIED FROM DigitalSreeni
https://github.com/bnsreenu/python_for_microscopists

CODE RAN ON SEPERATE MACHINE HENCE DIFFERENCE IN DIRECTORY

USES 15 IMAGES ADDED TO THE TRAINING DATA DIRECTORY
PROCESSES AND ADDS TO A NEW PARTITION
10 IMAGES FOR TRAINING
5 IMAGES FOR TESTING
SPLIT IS DONE MANUALLY

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
import sklearn
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler() # scalar obj

# helper lambda functions
scaleNiFTi = lambda img=None : scaler.fit_transform( img.reshape(-1, img.shape[1]) ).reshape(img.shape)
crop = lambda volume=None : volume[56:184, 56:184, 13:141] # 128x128x128 - we need a cube for unet 


# globals 
# original 17 removed since they've already been used, 15 added
training_path = 'C:\\Users\\nycdoe\\Desktop\\Ryan\\BraTS2020_TrainingData\\MICCAI_BraTS2020_TrainingData\\'
output_path = 'C:\\Users\\nycdoe\\Desktop\\Ryan\\Training_Processed2\\' #simple change


# get all volumnes
t1ce_paths = sorted(glob.glob(training_path + '*/*t1ce.nii')) # glob gets image by pathname that matches a pattern
t2_paths = sorted(glob.glob(training_path + '*/*t2.nii'))
flair_paths = sorted(glob.glob(training_path + '*/*flair.nii'))
segmentation_paths = sorted(glob.glob(training_path + '*/*seg.nii'))


# process each one
for img_num in range(12, 12+len(segmentation_paths)):
    
    print('Processing image ', img_num)

    # load images for current 
    t1ce = nb.loadsave.load(t1ce_paths[12 - img_num]).get_fdata()
    t2 = nb.loadsave.load(t2_paths[12 - img_num]).get_fdata()
    flair = nb.loadsave.load(flair_paths[12 - img_num]).get_fdata()
    segmentation = nb.loadsave.load(segmentation_paths[12 - img_num]).get_fdata()
    
    # scale the images from original intensities to 0-1
    t1ce = scaleNiFTi(t1ce)
    t2 = scaleNiFTi(t2)
    flair = scaleNiFTi(flair)
    
    # no need for float points for mask channel information
    # convert to 8 bit int
    segmentation = segmentation.astype(np.uint8)
    
    # replace 4 with 3 in the array since 3 is unused
    # will play an important role in create 3D UNet
    segmentation[segmentation == 4] = 3
    
    # now combine the different modals into 1 multi modal array
    combined = np.stack([flair, t1ce, t2], axis=3)
    combined = combined.astype(np.float16) # used for computational speed up 
    
    # crop combined multi-modal, and segmentation
    combined = crop(combined)
    segmentation = crop(segmentation)
    
    
    # recall that some of our images are completely useless, our data set is very small so this may not be necessary
    val, counts = np.unique(segmentation, return_counts=True)
    if (1 - (counts[0]/counts.sum())) > 0.01: # volumes contains 1% useful data for us o use towards training
        print(f'SAVING COMBINED...{img_num:4d}')
        
        # segmentation
        # PATCH COMBINED MODALITIES (COMMENT OUT IF PROCESSING VALIDATION DATA)
        
        seg_patches = []
        for hs in np.hsplit(segmentation, 2):
            
            for vs in np.vsplit(hs, 2):
                
                for ds in np.dsplit(vs, 2):
                    
                    # add a modality channel to the patch
                    # do this after the chop because we're adding
                    # another dim
                    ds = to_categorical(ds, num_classes=4) # categorical so we can index each modality
                    seg_patches.append(ds)
        
        #make sure segmentation has a modality channel!
        #segmentation = to_categorical(segmentation, num_classes=4)
        
        # because of my computers limitations
        # we'll apply a patch based method for data augmentation
        # and computational speed up
        # 8 77x77x77 patches for every volume
        
        patches = []
        for hs in np.hsplit(combined, 2):
            
            for vs in np.vsplit(hs, 2):
                
                for ds in np.dsplit(vs, 2):
                    patches.append(ds)
                    
        
        # decrease data type size for lowering computational complexity
        patches = np.array(patches)
        patches = np.astype(np.float32)
        seg_patches = np.array(seg_patches)
        seg_patches = seg_patches.astype(np.float32)
        
        # SAVE AS NUMPY ARRAYS TO AVOID UNNECESSARY IMAGES SAVING AND REINTERPRETATION
        for i, patch in enumerate(patches):
            np.save(output_path + 'training\\volumes\\volume_' + str(img_num) + '_patch_' + str(i) +'.npy', patch)
            np.save(output_path + 'training\\segmentations\\segmentation_' + str(img_num) + '_patch_' + str(i) +'.npy', seg_patches[i])
        
        
        # UNCOMMEENT IF PROCESSING VALIDATION DATA, OBVIOUSLY CHANGE PATHS
        # segmentation = to_categorical(segmentation, num_classes=4) # categorical so we can index each modality
        # segmentation = np.array(segmentation)
        # segmentation = segmentation.astype(np.float32)
        # combined = np.array(combined)
        # combined = combined.astype(float32)
        # for i, patch in enumerate(patches):
        #    np.save(output_path + 'training\\volumes\\volume_' + str(img_num) + '_patch_' + str(i) +'.npy', combined)
        #    np.save(output_path + 'training\\segmentations\\segmentation_' + str(img_num) + '_patch_' + str(i) +'.npy', segmentation)



# for testing purposes we'll split a majority of the training data
# into training and the rest into testing, we want to keep these seperate
# manually do this since splitfolders doesnt match our py version