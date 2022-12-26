#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 04:07:02 2022

@author: ryan
"""
from generator import load_batch
from matplotlib import pyplot as plt

#### load the model in for metrics ##################
from keras.models import load_model

my_model = load_model('brats_3d.hdf5', 
                      compile=False)

################################################


#### measure performance on testing data ##################

val_img_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/volumes/"
val_mask_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/val/segmentations/"

train_img_list= sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))

val_img_list= sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))

################################################

#### measure performance ##################

from keras.metrics import MeanIoU

batch_size=8 #Check IoU for a batch of images
test_img_datagen = load_batch(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

# generate a batch of images from numpy arrays
test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_mask_batch_argmax, test_pred_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

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

################################################

#### measure performance on testing data ##################

val_img_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/"
val_mask_dir = "/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/"

train_img_list= sorted(os.listdir(train_img_dir))
train_mask_list = sorted(os.listdir(train_mask_dir))

val_img_list= sorted(os.listdir(val_img_dir))
val_mask_list = sorted(os.listdir(val_mask_dir))

################################################

#### measure performance ##################

from keras.metrics import MeanIoU

batch_size=8 #Check IoU for a batch of images
test_img_datagen = load_batch(val_img_dir, val_img_list, 
                                val_mask_dir, val_mask_list, batch_size)

# generate a batch of images from numpy arrays
test_image_batch, test_mask_batch = test_img_datagen.__next__()

test_mask_batch_argmax = np.argmax(test_mask_batch, axis=4)
test_pred_batch = my_model.predict(test_image_batch)
test_pred_batch_argmax = np.argmax(test_pred_batch, axis=4)

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_mask_batch_argmax, test_pred_batch_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

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

################################################


#### predict on one image and visualize ########
img_num = random.randint(0, 9)
print("Validation image", img_num)
patch_num = 0

test_img = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/volumes/volume_"+str(img_num)+"_patch_"+str(patch_num)+".npy")

test_mask = np.load("/Users/ryan/Desktop/BIOMEDICAL IA/FINAL PROJECT/code/combined/validation/segmentations/segmentation_"+str(img_num)+"_patch_"+str(patch_num)+".npy")
test_mask_argmax=np.argmax(test_mask, axis=3)

test_img_input = np.expand_dims(test_img, axis=0)
test_prediction = my_model.predict(test_img_input)
test_prediction_argmax=np.argmax(test_prediction, axis=4)[0,:,:,:]

################################################


#### visualize ################################

#Plot individual slices from test predictions for verification
from matplotlib import pyplot as plt
import random

#n_slice=random.randint(0, test_prediction_argmax.shape[2])
n_slice = 55
plt.figure(figsize=(12, 8))
plt.subplot(231)
plt.title('Testing Image')
plt.imshow(test_img[:,:,n_slice,1], cmap='gray')
plt.subplot(232)
plt.title('Testing Label')
plt.imshow(test_mask_argmax[:,:,n_slice])
plt.subplot(233)
plt.title('Prediction on test image - 0')
plt.imshow(test_prediction_argmax[:,:, n_slice])
plt.subplot(234)
plt.title('Prediction on test image - 0')
plt.imshow(test_prediction[0][:,:, n_slice, 1])
plt.subplot(235)
plt.title('Prediction on test image - 0')
plt.imshow(test_prediction[0][:,:, n_slice, 2])
plt.subplot(236)
plt.title('Prediction on test image - 0')
plt.imshow(test_prediction[0][:,:, n_slice, 3])
plt.show()

n_classes = 4
IOU_keras = MeanIoU(num_classes=n_classes)  
IOU_keras.update_state(test_mask_argmax, test_prediction_argmax)
print("Mean IoU =", IOU_keras.result().numpy())

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
################################################