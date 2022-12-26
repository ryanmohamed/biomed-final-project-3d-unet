# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 02:14:01 2022

MAJORITY OF ALL CODE SOURCED OR MODIFIED FROM DigitalSreeni
https://github.com/bnsreenu/python_for_microscopists

PERFORM THE SECOND TRAINING ON THE UNWEIGHTED AND WEIGHT MODELS

@author: nycdoe
"""

import os
import numpy as np
from generator import load_batch
#import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import glob
import random

#################################
# find average frequency of each label in the ground truth segmentations of
# the training images
import pandas as pd
columns = ['0','1','2','3'] # our different labels, see paper
df = pd.DataFrame(columns=columns)

train_mask_list = sorted(glob.glob('C:\\Users\\nycdoe\\Desktop\\Ryan\\Training_Processed2\\' + 'training\\segmentations\\' + '*.npy'))
for img in range(len(train_mask_list)):
    print(img)
    temp_image=np.load(train_mask_list[img]) # load the img
    temp_image = np.argmax(temp_image, axis=3)
    val, counts = np.unique(temp_image, return_counts=True)
    zipped = zip(columns, counts)
    conts_dict = dict(zipped)
    
    df = df.append(conts_dict, ignore_index=True)

label_0 = df['0'].sum()
label_1 = df['1'].sum()
label_2 = df['1'].sum()
label_3 = df['3'].sum()
total_labels = label_0 + label_1 + label_2 + label_3
n_classes = 4

wt0 = round((total_labels/(n_classes*label_0)), 2) 
wt1 = round((total_labels/(n_classes*label_1)), 2)
wt2 = round((total_labels/(n_classes*label_2)), 2)
wt3 = round((total_labels/(n_classes*label_3)), 2)

print("Pre-evaluated weights")
print(wt0, wt1, wt2, wt3)
##############################################################



##############################################################
#Define the image generators for training and validation

train_img_dir = 'C:\\Users\\nycdoe\\Desktop\\Ryan\\Training_Processed2\\' + 'training\\volumes\\'
train_mask_dir = 'C:\\Users\\nycdoe\\Desktop\\Ryan\\Training_Processed2\\' + 'training\\segmentations\\'

val_img_dir = 'C:\\Users\\nycdoe\\Desktop\\Ryan\\Training_Processed2\\' + 'val\\volumes\\'
val_mask_dir = 'C:\\Users\\nycdoe\\Desktop\\Ryan\\Training_Processed2\\' + 'val\\segmentations\\'

train_img_list= sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))

val_img_list= sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))

##################################

########################################################################
batch_size = 2

train_img_datagen = load_batch(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = load_batch(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()

img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]
test_mask=np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')ed rc
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()


###########################################################################
# Define loss, metrics and optimizer to be used for training
# use weights assigned previously 
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = keras.optimizers.Adam(LR)
#######################################################################


#################################################
## continue to train the model
from keras.models import load_model

#Now, let us add the iou_score function we used during our initial training
my_model = load_model('brats_weighted_3d.hdf5', 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)})

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size

#Now all set to continue the training process. 
history3=my_model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=10,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )

my_model.save('brats_weighted2_3d.hdf5')
#################################################



##################################################################
# plot the training and validation IoU and loss at each epoch
loss = history3.history['loss']
val_loss = history3.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history3.history['accuracy']
val_acc = history3.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#################################################


##############################################################
#Define the image generators for training and validation

train_img_dir = 'C:\\Users\\nycdoe\\Desktop\\Ryan\\Training_Processed2\\' + 'training\\volumes\\'
train_mask_dir = 'C:\\Users\\nycdoe\\Desktop\\Ryan\\Training_Processed2\\' + 'training\\segmentations\\'

val_img_dir = 'C:\\Users\\nycdoe\\Desktop\\Ryan\\Training_Processed2\\' + 'val\\volumes\\'
val_mask_dir = 'C:\\Users\\nycdoe\\Desktop\\Ryan\\Training_Processed2\\' + 'val\\segmentations\\'

train_img_list= sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))

val_img_list= sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))

##################################

########################################################################
batch_size = 2

train_img_datagen = load_batch(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = load_batch(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
img, msk = train_img_datagen.__next__()

img_num = random.randint(0,img.shape[0]-1)
test_img=img[img_num]
test_mask=msk[img_num]
test_mask=np.argmax(test_mask, axis=3)

n_slice=random.randint(0, test_mask.shape[2])
plt.figure(figsize=(12, 8))

plt.subplot(221)
plt.imshow(test_img[:,:,n_slice, 0], cmap='gray')
plt.title('Image flair')
plt.subplot(222)
plt.imshow(test_img[:,:,n_slice, 1], cmap='gray')
plt.title('Image t1ce')
plt.subplot(223)
plt.imshow(test_img[:,:,n_slice, 2], cmap='gray')
plt.title('Image t2')
plt.subplot(224)
plt.imshow(test_mask[:,:,n_slice])
plt.title('Mask')
plt.show()
########################################################################


###########################################################################
# Define loss, metrics and optimizer to be used for training
# use weights assigned previously 
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.25, 0.25, 0.25, 0.25])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = keras.optimizers.Adam(LR)
#######################################################################


#################################################
## continue to train the model
from keras.models import load_model

#Now, let us add the iou_score function we used during our initial training
my_model = load_model('brats_3d_unweighted.hdf5', 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)})

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size

#Now all set to continue the training process. 
history=my_model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=10,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )

my_model.save('brats_3d_unweighted2.hdf5')
#################################################



##################################################################
# plot the training and validation IoU and loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(epochs, acc, 'y', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
#################################################