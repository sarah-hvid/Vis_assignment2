"""
A script that performs a logistic regression classification on either the mnist_784 or cifar10 dataset.
"""
# system tools
import os
import sys
import argparse

# data analysis
import numpy as np

# image processing
import cv2

# Import sklearn metrics
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.datasets import cifar10


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


def fit_model(X_train_scaled, y_train, X_test_scaled, y_test):
    args = parse_args()
    data_set = args['data']
    
    clf = LogisticRegression(penalty='none', 
                             tol=0.1, 
                             solver='saga',
                             multi_class='multinomial').fit(X_train_scaled, y_train)
    
    y_pred = clf.predict(X_test_scaled)
    
    if data_set == 'cifar10':
        
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
        cm = metrics.classification_report(y_test, y_pred, target_names = labels)
        print(cm)
        
        with open("output/lr_report_cifar10.txt", "w") as f:
            print(cm, file=f)
        
    if data_set == 'mnist_784':
        cm = metrics.classification_report(y_test, y_pred)
        print(cm)
        
        with open("output/lr_report_mnist784.txt", "w") as f:
            print(cm, file=f)
            
    else:
        print('error')
        
    return
 
    
def main():
    
    # check if arguments are provided
    script = sys.argv[0]
    if len(sys.argv) == 1: # no arguments, so print help message
        print("""Error: an input is required\nUsage: input -d flag followed by the dataset""")
        return
    
    # parse arguments
    args = parse_args()
    data_set = args['data']
    
    # model process
    if data_set == 'mnist_784':
        X, y = load_mnist()
        X_train_scaled, X_test_scaled, y_train, y_test = split_data(X, y)
        fit_model(X_train_scaled, y_train, X_test_scaled, y_test)
     
    elif data_set == 'cifar10':
        X_train_dataset, X_test_dataset, y_train, y_test = load_cifar()
        fit_model(X_train_dataset, y_train, X_test_dataset, y_test)
    
    else:
        print('Input is in the wrong format. Write mnist_784 for digits dataset and cifar10 for animal/vehicles dataset.')
    
if __name__ == '__main__':
    main()