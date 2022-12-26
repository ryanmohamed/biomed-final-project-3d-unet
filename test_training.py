#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

MAJORITY OF ALL CODE SOURCED OR MODIFIED FROM DigitalSreeni
https://github.com/bnsreenu/python_for_microscopists

TRAIN THE MODELS
PERFORM PREDICTIONS
RECORD METRICS

@author: ryan
"""


import os
# make sure we're in the right directory to access the modules we created
os.chdir('/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code')


import numpy as np
from generator import load_batch

import keras # for metrics and loading/training the model
from matplotlib import pyplot as plt # visualization
import glob # file retrieval
import random


from keras.metrics import MeanIoU # for average iou metrics


###### utility for printing iou score ##########################################
def print_iou_score(label, prediction, ground_truth, n_classes=4):
    
    print(label + ' IOU SCORES')
    print('####################')
    
    IOU_keras = MeanIoU(num_classes=n_classes)  
    IOU_keras.update_state(ground_truth, prediction)
    print("IoU =", IOU_keras.result().numpy())

    # determine iou for the individual labels
    # 4x4 array of the weights respective to each label
    values = np.array(IOU_keras.get_weights()).reshape(n_classes, n_classes)

    # digital sreenis approach to seperating the individual accuracies and inaccuracies for each label
    unlabeled = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[1,0]+ values[2,0]+ values[3,0])
    ncr_net = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[0,1]+ values[2,1]+ values[3,1])
    ed = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[0,2]+ values[1,2]+ values[3,2])
    et = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[0,3]+ values[1,3]+ values[2,3])

    print("Unlabeled IOU =", unlabeled)
    print("NCR/NET IOU =", ncr_net)
    print("ED IOU =", ed)
    print("ET IOU =", ncr_net)
    
def plot_test_and_prediction(test, ground_truth, prediction):
    n_slice = 55 # we'll keep these constant for consistency, always show the 55th z slice of an image
    plt.figure(figsize=(12, 8))
    
    plt.subplot(231)
    plt.title('Testing Image')
    plt.imshow(test[:,:,n_slice,1], cmap='gray')
    
    plt.subplot(232)
    plt.title('Testing Label')
    plt.imshow(ground_truth[:,:,n_slice])
    
    plt.subplot(233)
    plt.title('Prediction on test image')
    plt.imshow(prediction[:,:, n_slice])

    plt.show()
    

#############################################################
# find weights of each label 
# method provided like all others
import pandas as pd
columns = ['0','1', '2', '3']
df = pd.DataFrame(columns=columns)
train_mask_list = sorted(glob.glob('/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/training/segmentations/*.npy'))
for img in range(len(train_mask_list)):
    print(img)
    temp_image=np.load(train_mask_list[img])
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
#Class weights claculation: n_samples / (n_classes * n_samples_for_class)
wt0 = round((total_labels/(n_classes*label_0)), 2) #round to 2 decimals
wt1 = round((total_labels/(n_classes*label_1)), 2)
wt2 = round((total_labels/(n_classes*label_2)), 2)
wt3 = round((total_labels/(n_classes*label_3)), 2)

# will be used in our loss function, specifically the dice function
##############################################################




##############################################################
#Define the image generators for training and validation

train_img_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/training/volumes/"
train_mask_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/training/segmentations/"

val_img_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/"
val_mask_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/"

train_img_list= sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))

val_img_list= sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))
#############################################################



########################################################################
# use generator to generate a batch of images from the given directories
batch_size = 2

train_img_datagen = load_batch(train_img_dir, train_img_list, 
                                train_mask_dir, train_mask_list, batch_size)

val_img_datagen = load_batch(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

img, msk = train_img_datagen.__next__()
########################################################################


###########################################################################
#Define loss, metrics and optimizer to be used for training
# Use a combination of dice and categorical focal loss, reasoning in paper
# wt0, wt1, wt2, wt3 = 0.25,0.25,0.25,0.25 - UNCOMMENT TO OVERRIDE PRE-EVALUATED WEIGHTS
import segmentation_models_3D as sm
dice_loss = sm.losses.DiceLoss(class_weights=np.array([wt0, wt1, wt2, wt3])) 
focal_loss = sm.losses.CategoricalFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = ['accuracy', sm.metrics.IOUScore(threshold=0.5)]

LR = 0.0001
optim = keras.optimizers.Adam(LR)
#######################################################################
#Fit the model 

steps_per_epoch = len(train_img_list)//batch_size
val_steps_per_epoch = len(val_img_list)//batch_size


from  my_unet import simple_unet_model

model = simple_unet_model(IMG_HEIGHT=64, 
                          IMG_WIDTH=64, 
                          IMG_DEPTH=64, 
                          IMG_CHANNELS=3, 
                          num_classes=4)

model.compile(optimizer = optim, loss=total_loss, metrics=metrics)
print(model.summary())

print(model.input_shape)
print(model.output_shape)

history=model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=25,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )

model.save('brats_3d_weighted.hdf5') #re-run the above lines to save differents model

# CODE IS BROKEN INTO SECTIONS AND SOMETIMES RAN OUT OF ORDER
# IN THE SPYDER IDE, hence the reason for no other model creation code
##################################################################


#plot the training and validation IoU and loss at each epoch
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
from keras.models import load_model
 

#Now, let us add the iou_score function we used during our initial training
my_model = load_model('brats_3d_unweighted.hdf5', 
                      custom_objects={'dice_loss_plus_1focal_loss': total_loss,
                                      'iou_score':sm.metrics.IOUScore(threshold=0.5)})

#Now all set to continue the training process. 
history2=my_model.fit(train_img_datagen,
          steps_per_epoch=steps_per_epoch,
          epochs=1,
          verbose=1,
          validation_data=val_img_datagen,
          validation_steps=val_steps_per_epoch,
          )


##################################################
# LOAD IN THE UNWEIGHTED MODEL (25 EPOCHS)
# for prediction only, no need to compile

from keras.models import load_model
my_model = load_model('brats_3d.hdf5', 
                      compile=False)

##################################################


##################################################
# Predict on single volumes - TESTING
img_num = 16 
patch_num = 6 # validation data is not patched
print("Testing volume", img_num, ": Patch", patch_num)

# get validation volume and ground truth
test_vol = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/volumes/volume_"+str(img_num)+"_patch_"+str(patch_num)+".npy")
test_gt = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/segmentations/segmentation_"+str(img_num)+"_patch_"+str(patch_num)+".npy")

test_gt_argmax = np.argmax(test_gt, axis=3) # maximum values along axis

test_vol_expanded = np.expand_dims(test_vol, axis=0) # add another axis so it works with our model, basically put this in an array
test_prediction = my_model.predict(test_vol_expanded)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0,:,:,:]

# output of prediction and the ground truth
print(test_prediction_argmax.shape)
print(test_gt_argmax.shape)
print(np.unique(test_prediction_argmax)) # labels found in the prediction


# visualize the prediction
plot_test_and_prediction(test_vol, 
                         ground_truth=test_gt_argmax, 
                         prediction=test_prediction_argmax)

# get iou
print_iou_score('Unweighted (25 epochs - TESTING IMAGE '+ str(img_num) +')', 
                test_gt_argmax, 
                test_prediction_argmax,
                n_classes=4)
##################################################



##################################################
# Predict on single volumes - VALIDATION
img_num = 5 
patch_num = 0 # validation data is not patched
print("Validation volume", img_num, ": Patch", patch_num)

# get validation volume and ground truth
val_vol = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/volume_"+str(img_num)+"_patch_"+str(patch_num)+".npy")
val_gt = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/segmentation_"+str(img_num)+"_patch_"+str(patch_num)+".npy")

val_gt_argmax = np.argmax(val_gt, axis=3) # maximum values along axis

val_vol_expanded = np.expand_dims(val_vol, axis=0) # add another axis so it works with our model, basically put this in an array
val_prediction = my_model.predict(val_vol_expanded)
val_prediction_argmax = np.argmax(val_prediction, axis=4)[0,:,:,:]

# output of prediction and the ground truth
print(val_prediction_argmax.shape)
print(val_gt_argmax.shape)
print(np.unique(val_prediction_argmax)) # labels found in the prediction


# visualize the prediction
plot_test_and_prediction(val_vol, 
                         ground_truth=val_gt_argmax, 
                         prediction=val_prediction_argmax)

# get iou
print_iou_score('Unweighted (25 epochs - VALIDATION IMAGE '+ str(img_num) +')', 
                val_gt_argmax, 
                val_prediction_argmax,
                n_classes=4)
##################################################



#################################################
# BATCH TESTING UNWEIGHTED MODEL - TESTING DATA 

# get the list of validation data for batch-testing

test_vol_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/volumes/"
test_gt_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/segmentations/"

test_vol_list= sorted(os.listdir(test_vol_dir))
test_gt_list = sorted(os.listdir(test_gt_dir))


batch_size=8 #Check IoU for a batch of images
test_vol_datagen = load_batch(test_vol_dir, test_vol_list, 
                                test_gt_dir, test_gt_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
test_vol_batch, test_gt_batch = test_vol_datagen.__next__()

test_gt_batch_argmax = np.argmax(test_gt_batch, axis=4)
test_prediction_batch = my_model.predict(test_vol_batch)
test_prediction_batch_argmax = np.argmax(test_prediction_batch, axis=4)

print_iou_score('Unweighted (25 epochs - Batch size 8 - TESTING)', 
                test_gt_batch_argmax, 
                test_prediction_batch_argmax,
                n_classes=4)
#################################################



#################################################
# BATCH TESTING UNWEIGHTED MODEL - VALIDATION DATA 

# get the list of validation data for batch-testing
val_vol_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/"
val_gt_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/"

val_vol_list= sorted(os.listdir(val_vol_dir))
val_gt_list = sorted(os.listdir(val_gt_dir))


batch_size=8 #Check IoU for a batch of images
val_vol_datagen = load_batch(val_vol_dir, val_vol_list, 
                                val_gt_dir, val_gt_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
val_vol_batch, val_gt_batch = val_vol_datagen.__next__()

val_gt_batch_argmax = np.argmax(val_gt_batch, axis=4)
val_prediction_batch = my_model.predict(val_vol_batch)
val_prediction_batch_argmax = np.argmax(val_prediction_batch, axis=4)

print_iou_score('Unweighted (25 epochs - Batch size 8 - VALIDATION)', 
                val_gt_batch_argmax, 
                val_prediction_batch_argmax,
                n_classes=4)
##################################################



##################################################
# LOAD IN THE WEIGHTED MODEL (25 EPOCHS)
# for prediction only, no need to compile

from keras.models import load_model
my_model = load_model('brats_weighted_3d.hdf5', 
                      compile=False)

##################################################


##################################################
# Predict on single volumes - TESTING
img_num = 16 
patch_num = 6 # validation data is not patched
print("Testing volume", img_num, ": Patch", patch_num)

# get validation volume and ground truth
test_vol = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/volumes/volume_"+str(img_num)+"_patch_"+str(patch_num)+".npy")
test_gt = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/segmentations/segmentation_"+str(img_num)+"_patch_"+str(patch_num)+".npy")

test_gt_argmax = np.argmax(test_gt, axis=3) # maximum values along axis

test_vol_expanded = np.expand_dims(test_vol, axis=0) # add another axis so it works with our model, basically put this in an array
test_prediction = my_model.predict(test_vol_expanded)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0,:,:,:]

# output of prediction and the ground truth
print(test_prediction_argmax.shape)
print(test_gt_argmax.shape)
print(np.unique(test_prediction_argmax)) # labels found in the prediction


# visualize the prediction
plot_test_and_prediction(test_vol, 
                         ground_truth=test_gt_argmax, 
                         prediction=test_prediction_argmax)

# get iou
print_iou_score('Weighted (25 epochs - TESTING IMAGE '+ str(img_num) +')', 
                test_gt_argmax, 
                test_prediction_argmax,
                n_classes=4)
##################################################



##################################################
# Predict on single volumes - VALIDATION
img_num = 5 
patch_num = 0 # validation data is not patched
print("Validation volume", img_num, ": Patch", patch_num)

# get validation volume and ground truth
val_vol = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/volume_"+str(img_num)+"_patch_"+str(patch_num)+".npy")
val_gt = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/segmentation_"+str(img_num)+"_patch_"+str(patch_num)+".npy")

val_gt_argmax = np.argmax(val_gt, axis=3) # maximum values along axis

val_vol_expanded = np.expand_dims(val_vol, axis=0) # add another axis so it works with our model, basically put this in an array
val_prediction = my_model.predict(val_vol_expanded)
val_prediction_argmax = np.argmax(val_prediction, axis=4)[0,:,:,:]

# output of prediction and the ground truth
print(val_prediction_argmax.shape)
print(val_gt_argmax.shape)
print(np.unique(val_prediction_argmax)) # labels found in the prediction


# visualize the prediction
plot_test_and_prediction(val_vol, 
                         ground_truth=val_gt_argmax, 
                         prediction=val_prediction_argmax)

# get iou
print_iou_score('Weighted (25 epochs - VALIDATION IMAGE '+ str(img_num) +')', 
                val_gt_argmax, 
                val_prediction_argmax,
                n_classes=4)
##################################################



#################################################
# BATCH TESTING WEIGHTED MODEL - TESTING DATA 

# get the list of validation data for batch-testing

test_vol_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/volumes/"
test_gt_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/segmentations/"

test_vol_list= sorted(os.listdir(test_vol_dir))
test_gt_list = sorted(os.listdir(test_gt_dir))


batch_size=8 #Check IoU for a batch of images
test_vol_datagen = load_batch(test_vol_dir, test_vol_list, 
                                test_gt_dir, test_gt_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
test_vol_batch, test_gt_batch = test_vol_datagen.__next__()

test_gt_batch_argmax = np.argmax(test_gt_batch, axis=4)
test_prediction_batch = my_model.predict(test_vol_batch)
test_prediction_batch_argmax = np.argmax(test_prediction_batch, axis=4)

print_iou_score('Weighted (25 epochs - Batch size 8 - TESTING)', 
                test_gt_batch_argmax, 
                test_prediction_batch_argmax,
                n_classes=4)
#################################################



#################################################
# BATCH TESTING WEIGHTED MODEL - VALIDATION DATA 

# get the list of validation data for batch-testing
val_vol_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/"
val_gt_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/"

val_vol_list= sorted(os.listdir(val_vol_dir))
val_gt_list = sorted(os.listdir(val_gt_dir))


batch_size=8 #Check IoU for a batch of images
val_vol_datagen = load_batch(val_vol_dir, val_vol_list, 
                                val_gt_dir, val_gt_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
val_vol_batch, val_gt_batch = val_vol_datagen.__next__()

val_gt_batch_argmax = np.argmax(val_gt_batch, axis=4)
val_prediction_batch = my_model.predict(val_vol_batch)
val_prediction_batch_argmax = np.argmax(val_prediction_batch, axis=4)

print_iou_score('Weighted (25 epochs - Batch size 8 - VALIDATION)', 
                val_gt_batch_argmax, 
                val_prediction_batch_argmax,
                n_classes=4)
##################################################





##################################################
# LOAD IN THE 2ND TRAINED UNWEIGHTED MODEL (25 EPOCHS)
# for prediction only, no need to compile

from keras.models import load_model
my_model = load_model('brats_3d_unweighted2.hdf5', 
                      compile=False)

##################################################


##################################################
# Predict on single volumes - TESTING
img_num = 16 
patch_num = 6 # validation data is not patched
print("Testing volume", img_num, ": Patch", patch_num)

# get validation volume and ground truth
test_vol = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/volumes/volume_"+str(img_num)+"_patch_"+str(patch_num)+".npy")
test_gt = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/segmentations/segmentation_"+str(img_num)+"_patch_"+str(patch_num)+".npy")

test_gt_argmax = np.argmax(test_gt, axis=3) # maximum values along axis

test_vol_expanded = np.expand_dims(test_vol, axis=0) # add another axis so it works with our model, basically put this in an array
test_prediction = my_model.predict(test_vol_expanded)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0,:,:,:]

# output of prediction and the ground truth
print(test_prediction_argmax.shape)
print(test_gt_argmax.shape)
print(np.unique(test_prediction_argmax)) # labels found in the prediction


# visualize the prediction
plot_test_and_prediction(test_vol, 
                         ground_truth=test_gt_argmax, 
                         prediction=test_prediction_argmax)

# get iou
print_iou_score('Unweighted2 (35 epochs - TESTING IMAGE '+ str(img_num) +')', 
                test_gt_argmax, 
                test_prediction_argmax,
                n_classes=4)
##################################################



##################################################
# Predict on single volumes - VALIDATION
img_num = 5 
patch_num = 0 # validation data is not patched
print("Validation volume", img_num, ": Patch", patch_num)

# get validation volume and ground truth
val_vol = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/volume_"+str(img_num)+"_patch_"+str(patch_num)+".npy")
val_gt = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/segmentation_"+str(img_num)+"_patch_"+str(patch_num)+".npy")

val_gt_argmax = np.argmax(val_gt, axis=3) # maximum values along axis

val_vol_expanded = np.expand_dims(val_vol, axis=0) # add another axis so it works with our model, basically put this in an array
val_prediction = my_model.predict(val_vol_expanded)
val_prediction_argmax = np.argmax(val_prediction, axis=4)[0,:,:,:]

# output of prediction and the ground truth
print(val_prediction_argmax.shape)
print(val_gt_argmax.shape)
print(np.unique(val_prediction_argmax)) # labels found in the prediction


# visualize the prediction
plot_test_and_prediction(val_vol, 
                         ground_truth=val_gt_argmax, 
                         prediction=val_prediction_argmax)

# get iou
print_iou_score('Unweighted2 (35 epochs - VALIDATION IMAGE '+ str(img_num) +')', 
                val_gt_argmax, 
                val_prediction_argmax,
                n_classes=4)
##################################################



#################################################
# BATCH TESTING WEIGHTED MODEL - TESTING DATA 

# get the list of validation data for batch-testing

test_vol_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/volumes/"
test_gt_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/segmentations/"

test_vol_list= sorted(os.listdir(test_vol_dir))
test_gt_list = sorted(os.listdir(test_gt_dir))


batch_size=8 #Check IoU for a batch of images
test_vol_datagen = load_batch(test_vol_dir, test_vol_list, 
                                test_gt_dir, test_gt_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
test_vol_batch, test_gt_batch = test_vol_datagen.__next__()

test_gt_batch_argmax = np.argmax(test_gt_batch, axis=4)
test_prediction_batch = my_model.predict(test_vol_batch)
test_prediction_batch_argmax = np.argmax(test_prediction_batch, axis=4)

print_iou_score('Unweighted2 (35 epochs - Batch size 8 - TESTING)', 
                test_gt_batch_argmax, 
                test_prediction_batch_argmax,
                n_classes=4)
#################################################



#################################################
# BATCH TESTING WEIGHTED MODEL - VALIDATION DATA 

# get the list of validation data for batch-testing
val_vol_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/"
val_gt_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/"

val_vol_list= sorted(os.listdir(val_vol_dir))
val_gt_list = sorted(os.listdir(val_gt_dir))


batch_size=8 #Check IoU for a batch of images
val_vol_datagen = load_batch(val_vol_dir, val_vol_list, 
                                val_gt_dir, val_gt_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
val_vol_batch, val_gt_batch = val_vol_datagen.__next__()

val_gt_batch_argmax = np.argmax(val_gt_batch, axis=4)
val_prediction_batch = my_model.predict(val_vol_batch)
val_prediction_batch_argmax = np.argmax(val_prediction_batch, axis=4)

print_iou_score('Unweighted2 (35 epochs - Batch size 8 - VALIDATION)', 
                val_gt_batch_argmax, 
                val_prediction_batch_argmax,
                n_classes=4)
##################################################





##################################################
# LOAD IN THE 2ND TRAINED WEIGHTED MODEL (25 EPOCHS)
# for prediction only, no need to compile

from keras.models import load_model
my_model = load_model('brats_weighted2_3d.hdf5', 
                      compile=False)

##################################################


##################################################
# Predict on single volumes - TESTING
img_num = 16 
patch_num = 6 # validation data is not patched
print("Testing volume", img_num, ": Patch", patch_num)

# get validation volume and ground truth
test_vol = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/volumes/volume_"+str(img_num)+"_patch_"+str(patch_num)+".npy")
test_gt = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/segmentations/segmentation_"+str(img_num)+"_patch_"+str(patch_num)+".npy")

test_gt_argmax = np.argmax(test_gt, axis=3) # maximum values along axis

test_vol_expanded = np.expand_dims(test_vol, axis=0) # add another axis so it works with our model, basically put this in an array
test_prediction = my_model.predict(test_vol_expanded)
test_prediction_argmax = np.argmax(test_prediction, axis=4)[0,:,:,:]

# output of prediction and the ground truth
print(test_prediction_argmax.shape)
print(test_gt_argmax.shape)
print(np.unique(test_prediction_argmax)) # labels found in the prediction


# visualize the prediction
plot_test_and_prediction(test_vol, 
                         ground_truth=test_gt_argmax, 
                         prediction=test_prediction_argmax)

# get iou
print_iou_score('Weighted2 (35 epochs - TESTING IMAGE '+ str(img_num) +')', 
                test_gt_argmax, 
                test_prediction_argmax,
                n_classes=4)
##################################################



##################################################
# Predict on single volumes - VALIDATION
img_num = 5 
patch_num = 0 # validation data is not patched
print("Validation volume", img_num, ": Patch", patch_num)

# get validation volume and ground truth
val_vol = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/volume_"+str(img_num)+"_patch_"+str(patch_num)+".npy")
val_gt = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/segmentation_"+str(img_num)+"_patch_"+str(patch_num)+".npy")

val_gt_argmax = np.argmax(val_gt, axis=3) # maximum values along axis

val_vol_expanded = np.expand_dims(val_vol, axis=0) # add another axis so it works with our model, basically put this in an array
val_prediction = my_model.predict(val_vol_expanded)
val_prediction_argmax = np.argmax(val_prediction, axis=4)[0,:,:,:]

# output of prediction and the ground truth
print(val_prediction_argmax.shape)
print(val_gt_argmax.shape)
print(np.unique(val_prediction_argmax)) # labels found in the prediction


# visualize the prediction
plot_test_and_prediction(val_vol, 
                         ground_truth=val_gt_argmax, 
                         prediction=val_prediction_argmax)

# get iou
print_iou_score('Weighted2 (35 epochs - VALIDATION IMAGE '+ str(img_num) +')', 
                val_gt_argmax, 
                val_prediction_argmax,
                n_classes=4)
##################################################



#################################################
# BATCH TESTING WEIGHTED MODEL - TESTING DATA 

# get the list of validation data for batch-testing

test_vol_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/volumes/"
test_gt_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/segmentations/"

test_vol_list= sorted(os.listdir(test_vol_dir))
test_gt_list = sorted(os.listdir(test_gt_dir))


batch_size=8 #Check IoU for a batch of images
test_vol_datagen = load_batch(test_vol_dir, test_vol_list, 
                                test_gt_dir, test_gt_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
test_vol_batch, test_gt_batch = test_vol_datagen.__next__()

test_gt_batch_argmax = np.argmax(test_gt_batch, axis=4)
test_prediction_batch = my_model.predict(test_vol_batch)
test_prediction_batch_argmax = np.argmax(test_prediction_batch, axis=4)

print_iou_score('Weighted2 (35 epochs - Batch size 8 - TESTING)', 
                test_gt_batch_argmax, 
                test_prediction_batch_argmax,
                n_classes=4)
#################################################



#################################################
# BATCH TESTING WEIGHTED MODEL - VALIDATION DATA 

# get the list of validation data for batch-testing
val_vol_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/"
val_gt_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/"

val_vol_list= sorted(os.listdir(val_vol_dir))
val_gt_list = sorted(os.listdir(val_gt_dir))


batch_size=8 #Check IoU for a batch of images
val_vol_datagen = load_batch(val_vol_dir, val_vol_list, 
                                val_gt_dir, val_gt_list, batch_size)

#Verify generator.... In python 3 next() is renamed as __next__()
val_vol_batch, val_gt_batch = val_vol_datagen.__next__()

val_gt_batch_argmax = np.argmax(val_gt_batch, axis=4)
val_prediction_batch = my_model.predict(val_vol_batch)
val_prediction_batch_argmax = np.argmax(val_prediction_batch, axis=4)

print_iou_score('Weighted2 (35 epochs - Batch size 8 - VALIDATION)', 
                val_gt_batch_argmax, 
                val_prediction_batch_argmax,
                n_classes=4)
##################################################








