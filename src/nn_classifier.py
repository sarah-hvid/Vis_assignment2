"""
A script that performs a neural network classification on either the mnist_784 or the cifar10 dataset.
"""

# path tools
import sys, os
import argparse

# image processing
import cv2

# neural networks with numpy
import numpy as np
from tensorflow.keras.datasets import cifar10
from utils.neuralnetwork import NeuralNetwork

# machine learning tools
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression


# function that specifies the required arguments
def parse_args():
    # Initialise argparse
    ap = argparse.ArgumentParser()
    
    # command line parameters
    ap.add_argument("-d", "--data", required = True, help = "The dataset we want to work with, mnist_784 or cifar10")

    args = vars(ap.parse_args())
    return args


def load_mnist():
    
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = np.array(X)
    y = np.array(y)
    
    return X, y

def minmax_scaling(data):
    X_norm = (data - data.min()) / (data.max() - data.min())
    return X_norm


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                        y, 
                                                        random_state=9,
                                                        train_size=7500, 
                                                        test_size=2500)
    
    #scaling the features
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0
    
    return X_train_scaled, X_test_scaled, y_train, y_test


def load_cifar():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    
    # preprocessing the images
    X_train_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    X_test_grey = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
   
    # scaling
    X_train_scaled = minmax_scaling(X_train_grey)
    X_test_scaled = minmax_scaling(X_test_grey)
    
    # reshaping the data
    nsamples, nx, ny = X_train_scaled.shape
    X_train_dataset = X_train_scaled.reshape(nsamples, nx*ny)
    
    nsamples, nx, ny = X_test_scaled.shape
    X_test_dataset = X_test_scaled.reshape(nsamples, nx*ny)
    
    return X_train_dataset, X_test_dataset, y_train, y_test


def fit_model(X_train_dataset, X_test_dataset, y_train, y_test):
    args = parse_args()
    data_set = args['data']
    
    y_train = LabelBinarizer().fit_transform(y_train) 
    y_test = LabelBinarizer().fit_transform(y_test)

  
    #####################################
    print('[INFO] training network...')

    input_shape = X_train_dataset.shape[1]
    nn = NeuralNetwork([input_shape, 64, 10])

    print(f'[INFO] {nn}')

    nn.fit(X_train_dataset, y_train, epochs = 10, displayUpdate = 1)
    
    predictions = nn.predict(X_test_dataset)
    y_pred = predictions.argmax(axis=1)
    
    if data_set == 'cifar10':
        
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
        report = classification_report(y_test.argmax(axis = 1), y_pred, target_names = labels)
        print(report)
        
        with open("output/nn_report_cifar10.txt", "w") as f:
            print(report, file=f)
        
    if data_set == 'mnist_784':
        report = classification_report(y_test.argmax(axis = 1), y_pred)
        print(report)
        
        with open("output/nn_report_mnist784.txt", "w") as f:
            print(report, file=f)
            
    else:
        print('error')
        
    return


def main():
    
    # parse arguments
    args = parse_args()
    data_set = args['data']
    
    # model process
    if data_set == 'mnist_784':
        X, y = load_mnist()
        X_train_scaled, X_test_scaled, y_train, y_test = split_data(X, y)
        fit_model(X_train_scaled,  X_test_scaled, y_train, y_test)
     
    elif data_set == 'cifar10':
        X_train_dataset, X_test_dataset, y_train, y_test = load_cifar()
        print(y_train.shape)
        fit_model(X_train_dataset, X_test_dataset, y_train, y_test)
    
    else:
        print('Input is in the wrong format. Write mnist_784 for digits dataset and cifar10 for animal/vehicles dataset.')
    
    
    
if __name__ == '__main__':
    main()