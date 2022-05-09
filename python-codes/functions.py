# This work is licensed under a Creative Commons license CC BY 4.0
# Authors: S. Gaudez, M. Ben Haj Slama, A. Kaestner and M.V. Upadhyay

import os
import keras.metrics     as metrics
import keras.losses      as loss
import keras.optimizers  as opt
from keras.models        import Model
from keras.layers        import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, BatchNormalization, Activation
import numpy             as np
import nrrd
import scipy
from scipy               import ndimage


''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Building the U-net model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
def buildSpotUNet(base_depth = 8) :
    # input layer
    in_img = Input((None, None, None, 1), name='Image_Input')
    
    # Encoder part
    layer_1 = Conv3D(base_depth, kernel_size=(3, 3, 3), padding='same')(in_img)
    layer_1 = BatchNormalization()(layer_1)
    layer_1 = Activation('relu')(layer_1)
    
    layer_2 = Conv3D(base_depth, kernel_size=(3, 3, 3), padding='same')(layer_1)
    layer_2 = BatchNormalization()(layer_2)
    layer_2 = Activation('relu')(layer_2)
   
    layer_3 = MaxPooling3D(pool_size=(2, 2, 2))(layer_2)

    layer_4 = Conv3D(base_depth*2, kernel_size=(3, 3, 3), padding='same')(layer_3)
    layer_4 = BatchNormalization()(layer_4)
    layer_4 = Activation('relu')(layer_4)
    
    layer_5 = Conv3D(base_depth*2, kernel_size=(3, 3, 3), padding='same')(layer_4)
    layer_5 = BatchNormalization()(layer_5)
    layer_5 = Activation('relu')(layer_5)
  
    layer_6 = MaxPooling3D(pool_size=(2, 2, 2))(layer_5)

    layer_7 = Conv3D(base_depth*4, kernel_size=(3, 3, 3), padding='same')(layer_6)
    layer_7 = BatchNormalization()(layer_7)
    layer_7 = Activation('relu')(layer_7)

    layer_8 = Conv3D(base_depth*4, kernel_size=(3, 3, 3), padding='same')(layer_7)  
    layer_8 = BatchNormalization()(layer_8)
    layer_8 = Activation('relu')(layer_8)

    layer_9 = MaxPooling3D(pool_size=(2, 2, 2))(layer_8)

    layer_10 = Conv3D(base_depth*8, kernel_size=(3, 3, 3), padding='same')(layer_9)
    layer_10 = BatchNormalization()(layer_10)
    layer_10 = Activation('relu')(layer_10)

    layer_11 = Conv3D(base_depth*8, kernel_size=(3, 3, 3), padding='same')(layer_10)  
    layer_11 = BatchNormalization()(layer_11)
    layer_11 = Activation('relu')(layer_11)
 
    # Decoder part
    layer_12 = UpSampling3D((2, 2, 2))(layer_11)
    layer_13 = concatenate([layer_8, layer_12])
    
    layer_14 = Conv3D(base_depth*4, kernel_size=(3, 3, 3), padding='same')(layer_13)
    layer_14 = BatchNormalization()(layer_14)
    layer_14 = Activation('relu')(layer_14)
    
    layer_15 = Conv3D(base_depth*4, kernel_size=(3, 3, 3), padding='same',)(layer_14)
    layer_15 = BatchNormalization()(layer_15)
    layer_15 = Activation('relu')(layer_15)
    
    layer_16 = UpSampling3D((2, 2, 2))(layer_15)
    layer_17 = concatenate([layer_5, layer_16])
    
    layer_18 = Conv3D(base_depth*2, kernel_size=(3, 3, 3), padding='same')(layer_17)
    layer_18 = BatchNormalization()(layer_18)
    layer_18 = Activation('relu')(layer_18)
      
    layer_19 = Conv3D(base_depth*2, kernel_size=(3, 3, 3), padding='same')(layer_18)
    layer_19 = BatchNormalization()(layer_19)
    layer_19 = Activation('relu')(layer_19)

    layer_20 = UpSampling3D((2, 2, 2))(layer_19)
    layer_21 = concatenate([layer_2, layer_20])
    
    layer_22 = Conv3D(base_depth, kernel_size=(3, 3, 3), padding='same')(layer_21)
    layer_22 = BatchNormalization()(layer_22)
    layer_22 = Activation('relu')(layer_22)
      
    layer_23 = Conv3D(base_depth, kernel_size=(3, 3, 3), padding='same')(layer_22)
    layer_23 = BatchNormalization()(layer_23)
    layer_23 = Activation('relu')(layer_23)

    layer_24 = Conv3D(1, kernel_size=(1, 1, 1), padding='same', activation='sigmoid')(layer_23)
    
    # output layer
    t_unet = Model(inputs = [in_img], outputs = [layer_24], name = 'U-net_model')
    
    return t_unet
    
    
   
''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pre-processing image functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''
# takes as input the raw data and returns the normalized data (Z-score normalization)
def prep_img(x):
    return (x-x.mean())/x.std()


''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Functions to read images and masks and build the data set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

# function used to read and prepare the input data for the model training and validation.
# Data are loaded and can be divided into sub-volumes to fit the GPU available memory.
# Data augmentation can be done by adding rotation of the sub-volume.


nb_subVol_x = 5						# define the number of sub-volumes along x axis
nb_subVol_y = 5						# define the number of sub-volumes along y axis
length_subVol_x = 192					# define the length of sub-volumes along x axis
length_subVol_y = 192					# define the length of sub-volumes along y axis
nb_rotation = 1						# define the number of rotation, 1 means no rotation

rotation = [(0,1),(1,2),(0,2)]				# defined rotation axis

def readDataMask(fileRoot, fileName):
    mask = []
    for i in range(0,len(fileName)):
        name = fileRoot + fileName[i]								# create the input file name
        inputImage, header = nrrd.read(name)							# read the data
        print('Mask name : ', fileName[i])							# print the file name
        for k in range(0,nb_subVol_x) :							        # loop to sub-divide region of 192x192, 5 times along the x axis
            x1 = k*length_subVol_x
            x2 = (k+1)*length_subVol_x
            for l in range(0,nb_subVol_y) :							# loop to sub-divide region of 192x192, 5 times along the y axis
                y1 = l*length_subVol_y
                y2 = (l+1)*length_subVol_y
                c_inputImage = inputImage[x1:x2,y1:y2,:]
                for j in range(0,nb_rotation) :						        # loop allowing to do data augmentation trough rotation 
                    finputImage = scipy.ndimage.rotate(c_inputImage, 90, axes=rotation[j])	# rotation of the data
                    mask.append(np.expand_dims((finputImage[:,:]==0),-1))			# store the data into an array
    return np.array(mask)									# return the array


def readDataImage(fileRoot, fileName):
    image = []
    for i in range(0,len(fileName)):
        name = fileRoot + fileName[i]								# create the input file name
        inputImage, header = nrrd.read(name)							# read the data
        d_inputImage = inputImage.astype('float32')						# convert into 32 bits file
        print('Image name : ', fileName[i])							# print the file name
        for k in range(0,nb_subVol_x) :							        # loop to sub-divide region of 192x192, 5 times along the x axis
            x1 = k*length_subVol_x
            x2 = (k+1)*length_subVol_x
            for l in range(0,nb_subVol_y) :							# loop to sub-divide region of 192x192, 5 times along the y axis
                y1 = l*length_subVol_y
                y2 = (l+1)*length_subVol_y
                c_inputImage = d_inputImage[x1:x2,y1:y2,:]
                for j in range(0,nb_rotation):						        # loop allowing to do data augmentation trough rotation
                    finputImage = scipy.ndimage.rotate(c_inputImage, 90, axes=rotation[j])	# rotation of the data
                    image.append(np.expand_dims(prep_img(finputImage),-1))			# store the data into an array
    return np.array(image)									# return the array


    
''' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Function to read image to predict
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ '''

# function used to read and prepare the data to be predicted by the model.
# the function takes as input parameters: the file name and (x, y, z) coordinates of the region of interest.
# with the present hardware and software specifications, a data set of 1024x1024x256 voxels can be predicted

def readDataToPredict(fileName, x1, x2, y1, y2, z1, z2):
    image = []
    name = fileName
    inputImage, header = nrrd.read(name)							# read the data
    d_inputImage = inputImage.astype('float32')   						# convert into 32 bits file
    finputImage = d_inputImage[x1:x2,y1:y2,z1:z2]						# reduce the input data to the ROI
    image.append(np.expand_dims(prep_img(finputImage),-1))					# store the data into an array
    return np.array(image) 									# return the array

