# This work is licensed under a Creative Commons license CC BY 4.0
# Authors: S. Gaudez, M. Ben Haj Slama, A. Kaestner and M.V. Upadhyay

import os
import numpy            as np
import nrrd
from numpy import save
from numpy import load

from functions 	import *


# functions to read the images and masks for the model training and validation
# corresponding image and mask have the same increment (e.g., data file: data_01.nrrd and segmented file: segm_01.nrrd)
# the images are normalized and converted into a one dimensional array.


# File roots
fileRootTrainingImage = "../inputData/trainingImages/"
fileRootTrainingMask = "../inputData/trainingMasks/"
fileRootValidateImage = "../inputData/validateImages/"
fileRootValidateMask = "../inputData/validateMasks/"


# Load, transform and save training images
fileNameTrainingImage = sorted(os.listdir(fileRootTrainingImage))              # load the image file name and sort them
train_img = readDataImage(fileRootTrainingImage, fileNameTrainingImage)        # return the data set
save('../inputData/pythonFiles/trainingDataImage.npy', train_img)              # save the data set into a .npy file

del train_img                                                                  # delete the data set from the memory

# Load, transform and save training masks
fileNameTrainingMask = sorted(os.listdir(fileRootTrainingMask))
train_mask = readDataMask(fileRootTrainingMask, fileNameTrainingMask)
save('../inputData/pythonFiles/trainingDataMask.npy', train_mask)

del train_mask

# Load, transform and save validation images
fileNameValidateImage = sorted(os.listdir(fileRootValidateImage)) 
valid_img = readDataImage(fileRootValidateImage, fileNameValidateImage) 
save('../inputData/pythonFiles/validateDataImage.npy', valid_img)

del valid_img

# Load, transform and save validation masks
fileNameValidateMask = sorted(os.listdir(fileRootValidateMask))
valid_mask = readDataMask(fileRootValidateMask, fileNameValidateMask)
save('../inputData/pythonFiles/validateDataMask.npy', valid_mask)

del valid_mask


