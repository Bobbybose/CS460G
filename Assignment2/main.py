# Author: Bobby Bose
# Assignment 2: Linear and Polynomial Regression


# Imports
from re import A
import numpy as np
#import math
import pandas as pd
#pd.options.display.max_rows = 1000
#import matplotlib.pyplot as plt

def main():

    wine_labels = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
                   "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", 
                   "pH", "sulphates", "alcohol", "quality"
                  ]
    
    wine_features = wine_labels.copy().remove("quality")

    wine_data_df = pd.read_csv("datasets/winequality-red.csv", delimiter = ",", names = wine_labels)


    # Preprocessing input features
    wine_data_processed = wine_data_df.copy()
    for feature in wine_features:
        max = wine_data_processed[feature].max()
        min = wine_data_processed[feature].min()

        for index, row in wine_data_processed.iterrows():
            row[feature] = (row[feature] - min) / (max - min)

    weights, MSE = multiple_linear_regression(wine_data_processed, wine_features, "quality")

    return 1

# main()


def multiple_linear_regression(dataset, features, class_label):
    weights = []
    alpha = 0.5
    epochs = 0
    MSE = 0

    # Randomly initialize the weights
    weights = np.random.uniform(0, 20, len(features) + 1).tolist()

    if epochs != 1000:
        epochs += 1

        for index in range(len(weights)):
            gradient_total = 0

            for data in dataset:

                if index == 0:
                    gradient_total += (get_equation_prediction(weights, data) - data[class_label]) * 1
                else:
                    gradient_total += (get_equation_prediction(weights, data) - data[class_label]) * data[index-1]

            weights[index] -= alpha * 1/(len(dataset)) * gradient_total
    
    
    for data in dataset:
        MSE += ((get_equation_prediction(weights, data) - data[class_label]))**2
    MSE = 1/MSE

    return weights, MSE

# multiple_linear_regression()


def get_equation_prediction(weights, data):
    result = []
    
    for index in range(len(weights)):
        
        if index == 0:
            result.append(weights[index]*1)
        else:
            result.append(weights[index]*data[index])
    
    return result
# get_equation_prediction()