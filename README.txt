/*
* Filename:  README.txt
* Date:      04/12/2022
* Author:    Rutvij Shah
* Email:     rutvij.shah@utdallas.edu
* Course:    CS6384 Spring 2022
* Version:   1.0
* Copyright: 2022, All Rights Reserved
*
* Description: Approach and steps to run the model testing code.
*
*/

Project Requirements:
  - Tensorflow 2.0+
  - Python 3.8+
  - Pandas, Numpy (the usual suspects)

The proj2.ipynb file contains the training code as outlined below and has a few additional dependencies.


Approach:
  - Using tensorflow-metal on MacOS, the model was trained on an AMD GPU.

  - Given data was divided into train, validation and test sets (80, 10, 10).

  - The base model chosen was InceptionV3, which was trained on inaturalist dataset in addition to the imagenet dataset
    all models are trained on. The inaturalist dataset consists of images of various species of animals, plants
    and other natural objects. This makes it a good base model to specialize on images of flowers.

  - On top of the feature-vector producing InceptionV3 base model, a dense ReLU layer was added with a dropout layer of .4.

  - This layer adds further trainable parameters to enable transfer learning. The last layer was set up to be a
    classification softmax layer, giving the probability of an image 'x' belonging to one of the 13 classes.

  - The model was trained for 20 epochs, followed by a low-learning rate fine-tuning run of 3 epochs which adjusted the
    weights of the underlying model

  - Final training accuracy was 97.1%, validation accuracy was 95.7% and test accuracy was 98.6%

  - Aggregate accuracy on the entire dataset is 97.5%.


Testing:
The testing script takes the following inputs

    usage: proj2_test.py [-h] [-m MODEL] [-csv TEST_CSV] [-spl]

    Transfer Learning Test

    options:
    -h, --help
            show this help message and exit
    -m MODEL, --model MODEL
            Saved model
    -csv TEST_CSV, --test_csv TEST_CSV
            CSV file with true labels, header must be {image_path, label}
    -spl, --show-prediction-labels
            Show a table of actual and predicted labels

The default path to model is a relative path to the model in trained_models folder present within the zip file,
and points to the inception_inaturalist_flowers model.

User needs to provide the path to a test csv file unless it is name as flowers_test.csv

If the "-spl" flag is present, in addition to the model accuracy, each file, its label and predicted label are printed as a table.

