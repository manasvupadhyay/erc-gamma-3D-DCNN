# This work is licensed under a Creative Commons license CC BY 4.0
# Authors: S. Gaudez, M. Ben Haj Slama, A. Kaestner and M.V. Upadhyay

import os
import numpy             as np
import nrrd
import tensorflow        as tf
import keras.metrics     as metrics
import keras.losses      as loss
import keras.optimizers  as opt
from keras.models        import Model, Sequential
from keras.layers        import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from keras.models        import load_model, save_model
from keras.callbacks     import ModelCheckpoint, CSVLogger
from numpy		 import load
from functions		 import *


#os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Uncomment to run on CPU

'''
~~~~~~~~~~~ Read and build the data set for training and validation ~~~~~~~~~~~~
'''

train_img = load('../inputData/pythonFiles/trainingDataImage.npy')
train_mask = load('../inputData/pythonFiles/trainingDataMask.npy')
valid_img = load('../inputData/pythonFiles/validateDataImage.npy')
valid_mask = load('../inputData/pythonFiles/validateDataMask.npy')

print("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("Number of training data : ", np.shape(train_img))
print("Number of validate data : ", np.shape(valid_img))
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

weightFile = "modelAndWeights/current-best_modelWeight_val-loss.h5"
weightFileTrain = "modelAndWeights/current-best_modelWeight_val-accuracy.h5"
metricsFile = "metrics/metrics.log"
modelFile = "modelAndWeights/current-uNet_model.h5"


learningRate = 0.0050      # Control the rate at which an algorithm updates the parameter estimates or learns the values of the parameters
Nepochs = 200              # Number of iteration on the whole input data to train the model
batchSize = 2              # Number of data set for each batch


'''
~~~~~~~~~~~~~~~ U-net model building and print the model ~~~~~~~~~~~~~~~
'''


t_unet = buildSpotUNet(base_depth=8)
t_unet.summary() 								# Prints a summary of the network, layers and number of parameters

# Compile the neural network model
t_unet.compile(
    loss = tf.keras.losses.BinaryCrossentropy(), 				# define the loss function used (binary cross entropy as the problem resumes to a binary classification)
    optimizer = tf.keras.optimizers.Adam(learning_rate = learningRate), 	# function used to optimize the parameter with backpropagation based on the loss value
    metrics = [tf.keras.metrics.BinaryAccuracy(),
                tf.keras.metrics.TruePositives(name='tp'),
                tf.keras.metrics.TrueNegatives(name='tn'),
                tf.keras.metrics.FalsePositives(name='fp'),
                tf.keras.metrics.FalseNegatives(name='fn')]  		        # define the metrics to follow and record
)


# 'ModelCheckpoint' save the model weights if the monitored value was improved
# 'CSVLogger' save the metrics in a file
# Options to save at each epoch the weights if the validation loss has decreased
checkpoint = ModelCheckpoint(weightFile, monitor='val_loss', mode='min', save_best_only=True, save_freq='epoch', verbose=1)

# Options to save at each epoch the weights if the validation binary accuracy has decreased
checkpoint2 = ModelCheckpoint(weightFileTrain, monitor='val_binary_accuracy', mode='max', save_best_only=True, save_freq='epoch', verbose=1)

# Options to save the metrics at each epoch
logger = CSVLogger(metricsFile, separator=" ")


# Load previous weight if the model was already trained
# t_unet.load_weights("best_modelWeight_val-accuracy.h5")


'''
~~~~~~~~~~~~~~~~~~~~~~~~~~ Model training ~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
# Train the model with input data, defined model and parameters
history = t_unet.fit(train_img2, train_mask2,                                # training data set with images and masks
                          validation_data = (valid_img2, valid_mask2),       # validation data set with images and mask
                          epochs = Nepochs,                                  # number of epochs to train the model
                          batch_size = batchSize,                            # number of samples per batch -number of batch : Nsample/batchSize-
                          callbacks = [checkpoint, checkpoint2, logger] ,    # save the model when a monitored parameter is improved and save the metrics at each epoch
                          verbose = 1)                                       # terminal output to follow the training progression



'''
~~~~~~~~~~~~~~~~~~~~~~~~~~ Save the files ~~~~~~~~~~~~~~~~~~~~~~~~~~~
'''
t_unet.save(modelFile)

# print the saved files names with the model, fitted weights and metrics
print("\n ~~~~~~~~~~~~~~ Training done ~~~~~~~~~~~~~~~")
print("Model file", modelFile)
print("Weights file ", weightFile)
print("Metrics file ", metricsFile)
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n")

del t_unet



