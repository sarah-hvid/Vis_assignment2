# Assignment 2 - Image classifier benchmark scripts
 
 Link to github of this assignment: https://github.com/sarah-hvid/Vis_assignment2

## Assignment description
In this assignment two different approaches should be used to make classification predictions on image data:
- The first approach is a simple ```LogisticRegression()``` classifier
- The second approach trains a simple ```Neural Network class``` written in ```numpy``` 
The full assignment description is available in the ```assignment2.md``` file. 

## Methods
This problem relates to classification of images. Initially, the user must specify whether to work with the ```mnist_784``` data or the ```cifar10``` data. The ```mnist_784``` data is split into a training and test set. The ```cifar10``` data is scaled and reshaped.

- The logistic regression approach:
  - The model is initiated with ......  The data is then classified using multinomial logistic regression. The classification report is saved in the output folder. 
  
- The neural network approach:
  - The labels of the datasets are initially binarized. The script uses the premade module in ```neuralnetwork.py``` located in the ```src/utils``` folder. The neural network is then initialized with 10 epochs. The classification report is saved in the output folder. 

## Usage
In order to run the scripts, certain modules need to be installed. These can be found in the ```requirements.txt``` file. The folder structure must be the same as in this GitHub repository (ideally, clone the repository). The current working directory when running the script must be the one that contains the ```data```, ```output``` and ```src``` folder. Examples of how to run the scripts from the command line: 

The ```logistic_regression.py``` script:
- python src/logistic_regression.py -d mnist_784
- python src/logistic_regression.py -d cifar10

The ```nn_classifier.py``` script:
- python src/nn_classifier.py -d mnist_784
- python src/nn_classifier.py -d cifar10
  
Examples of the outputs of the scripts may be seen in the ```output``` folder. 

## Results
The logistic regression approach achieves quite different results across the two datasets. For the simpler ```mnist_784``` digits dataset it achieves a high accuracy score of 0.90. However, on the more complex ```cifar10``` dataset of animals and vehicles, the models only obtains an accuracy score of 0.31. While this is an accuracy better than chance, the results should be better to be useful. The simple logistic regression is therefore best suited for the ```mnist_784``` dataset.
The neural network approach achieved an accuracy of 0.93 on the ```mnist_784``` dataset. On the ```cifar10``` dataset it achieved an accuracy of 0.37. The results are therefore similar for either of the two approaches. However, it should be noted that it is a simple neural network written in ```numpy```.
