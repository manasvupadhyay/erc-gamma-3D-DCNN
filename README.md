# 3D deep convolutional neural network for precipitates and porosities segmentation tomograms (3D-DCNN U-net model)


*3D deep convolutional neural network segmentation model for precipitates and porosities identification in synchrotron X-ray tomograms*

*S. Gaudez, M. Ben Haj Slama, A. Kaestner and M.V. Upadhyay*

This work is licensed under a Creative Commons license CC BY 4.0

3D-DCNN model, experimental and segmented data presented in this work can be dowload (.zip file):
**[3D-DCNN](https://mycore.core-cloud.net/index.php/s/Ykz9uL5o0rv9bSz)**

## Model and implementation
This is an implementation of a 3D U-net model for Keras and Tensorflow in Python.
It is implemented in Python version 3.8.5 using:
- Keras python library version 2.4.3
- Tensorflow library version 2.5.0
- Numpy python library version 1.19.5
- Scipy python library version 1.3.3
- Nrrd python library version 0.4.2

To speed up the computations, the code was executed on a graphics processing unit (GPU) from NVIDIA.
To that end, we used:
- Cuda version 11.2
- cuDNN library version 8.2.1


## Model aim
The model has been developed to segment precipitates and porosities from synchrotron transmission X-ray micrography experimental data.
It was trained, validated and tested with manually labeled volumetric data (corresponding to 1280 slices).


## Organization
The model is organized in four different python codes:
1) *prepareData.py* is used to normalize and divide the experimental data into sub-volumes and to prepare the files (.npy files) for the training and validation steps of the model,
2) *trainingAndValidation.py* is used to train and validate the model,
3) *predictData.py* is used to make a prediction (i.e., to segment) from experimental data,
4) *functions.py* contains all the functions used in the previous Python files.


#### *prepareData.py*
>
> The code loads stored data in the trainingImages, trainingMasks, validationImages and validationMask folders.
> The data in these folders are the experimental data and their corresponding segmentations which will be used for the training and validation steps.
> Then, the data are normalized and divided into sub-volumes to fit the GPU available memory and then saved in .npy files in the pythonFiles folder.
> In order to do so, use the following command: python3 prepareData.py


#### *trainingAndValidation.py*

> .npy files are loaded to train and validate the model.
> Output files are generated during the training and validation:
> 1. Statistic parameters evolution as a function of the epoch.
> 2. Model architecture and fitted weights.
> 3. The last best fitted weights if the validation loss function has been decreased.
> 4. The last best fitted weights if the general validation accuracy has been increased.
> In order to un training, use the following command: python3 trainingAndValidation.py


#### *predictData.py*

> Model architecture and fitted weights are loaded as well as the experimal data to segment.
> It returns a file with the probability of each voxel belonging to a class (either precipitate or matrix).
> The data used to test the model in the manuscript are copied into the *test-file* folder. To test the model, the data inside this folder can be used.
> In order to do so, use the following command: python3 predictData.py
> Model and fitted weights are available in the *modelAndWeights/final-uNet_model.h5* and *modelAndWeights/final-trained-weights.h5* files, respectively.



For further information, please refer to the manuscript:

*3D deep convolutional neural network segmentation model for precipitates and porosities identification in synchrotron X-ray tomograms*
