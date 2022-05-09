# This work is licensed under a Creative Commons license CC BY 4.0
# Authors: S. Gaudez, M. Ben Haj Slama, A. Kaestner and M.V. Upadhyay

import os
import numpy             as np
import tensorflow        as tf
from keras.models        import Model
from keras.layers        import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, Activation
from keras.models        import load_model

import nrrd

from functions		 import *
### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
################################################################################################

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Uncomment to run on GPU


''' input image '''
# Define the input file name
nameInput = "../test-file/test-file_data.nrrd"
# Define the output file name
nameOutput = "../test-file/test-file_segmented.nrrd"


'''
~~~~~~~~~~~~~~~~~~~~~~ Size of the ROI investigated ~~~~~~~~~~~~~~~~~~~~~~~
'''
# these parameters have to be adjusted according to your region of interest and hardware specifications
x1 = 0
x2 = 1024
y1 = 0
y2 = 1024
z1 = 0
z2 = 128

# ROI defined above is loaded from the data to be predicted
dataInput = readDataToPredict(nameInput, x1, x2, y1, y2, z1, z2)
a, x, y, z, b = np.shape(dataInput)

# The machine learning model and pre-trained weights are loaded
t_unet = load_model('modelAndWeights/final-uNet_model.h5')
t_unet.load_weights('modelAndWeights/final-trained-weights.h5')

# Segmentation of the ROI defined using the machine learning model and fitted weights
unet_pred = t_unet.predict(dataInput)[0, :,  :, :,0]

# Save the predicted data into an .nrrd file
nrrd.write(nameOutput, unet_pred)

# print the size of the data segmented and file name
print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
print("Data volume : ", x, "x", y, "x", z, "pixels")
print("Output file : ", nameOutput)
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')

