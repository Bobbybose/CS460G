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

    wine_features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
                   "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", 
                   "pH", "sulphates", "alcohol"
                    ]

    wine_data_df = pd.read_csv("datasets/winequality-red.csv", delimiter = ",")#, names = wine_labels)

    # Preprocessing input features
    wine_data_processed = wine_data_df.copy()
    for feature in wine_features:
        max = wine_data_processed[feature].max()
        min = wine_data_processed[feature].min()

        for index, row in wine_data_processed.iterrows():
            row[feature] = (int(row[feature]) - min) / (max - min)

    weights, MSE = multiple_linear_regression(wine_data_processed, wine_features, "quality")

    print(weights)

    print(MSE)


    return 1

# main()


def multiple_linear_regression(dataset, features, class_label):
    weights = []
    alpha = 0.001
    epochs = 100
    MSE = 0

    # Randomly initialize the weights
    weights = np.random.uniform(0, 1, len(features) + 1).tolist()

    for epoch in range(epochs):

        if epoch % 10 == 0:
            print(weights)
            print(get_MSE(dataset, weights, class_label))

        for index in range(len(weights)):
            gradient_total = 0
            
            
            for i, data in dataset.iterrows():
                if index == 0:
                    gradient_total += (get_equation_prediction(weights, data) - data[class_label]) * 1
                else:
                    gradient_total += (get_equation_prediction(weights, data) - data[class_label]) * data[index-1]

            weights[index] -= alpha * 1/(len(dataset)) * gradient_total
    
    MSE = get_MSE(dataset, weights, class_label)

    return weights, MSE

# multiple_linear_regression()


def get_MSE(dataset, weights, class_label):
    MSE = 0
    
    for i, data in dataset.iterrows():
        MSE += ((get_equation_prediction(weights, data) - data[class_label]))**2
    MSE *= 1/len(dataset)

    return MSE

def get_equation_prediction(weights, data):
    result = 0
    for index in range(len(weights)):
        
        if index == 0:
            result += weights[index]*1
        else:
            result += weights[index]*data[index-1]
    
    return result
# get_equation_prediction()

main()