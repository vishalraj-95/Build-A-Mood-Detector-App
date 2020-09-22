# Build-A-Mood-Detector-App
Spotle AI-Thon Level II 
We noticed that there is a data imbalance among the classes. To counter the possibility of over-fitting towards the dominant class, we used "class weights" and image data augmentation. Keras image data generator was used for augmentation. The augmentation procedures like zoom, rotation etc. are used an its parameters have been tuned to reduce over-fitting



## Pre-processing: 



The data was normalized by scaling pixel values in the range [0,1], reshaped to 48x48x1(gray-scale). Labels are emotion wise label encoded.

We made a stratified train-test split to train the model on the training set and evaluate the model performance on the validation set. 



Since the data size was small (48x48), a deeper network could lead to over-fitting and learning related problems (exploding gradients, vanishing gradients). So we used a relatively smaller model:



## Model architecture:



1. 4 CNN blocks: each block consists of a convolutional layer with relu as activation, followed by batch norm and a max pooling layer 

2. The resulting feature vector is fed to a fully connected layer with swish as activation function 

3. The final layer consists of 3 neurons with softmax as activation

4. Optimizer used is "Adam" and loss function = categorical cross-entropy



We further tune our hyperparameters and learning rates to have efficient optimization



A deep Convolution Neural Network was used to build the mood detector app

The API is defined as follows:

def aithon_level2_api(trainingcsv, testcsv):

  model = cls.train_a_model(trainingcsv)
  result = cls.test_the_model(testcsv, model)
  return result


The train_a_model method is used to build our model.
The cnn_model() defines our Deep CNN.
The train_a_model(images) method is used to make the images and labels fit for out neural net and divide the data into testing and validation batches and used to train our model on the test images.
This method returns the model which is passed to testing the test dataset.

The test_the_model method is used to test our model.
test_the_model(testfile) to pre-process the images and make predictions on the test data using the our trained model.
This method returns the list of predicted labels
