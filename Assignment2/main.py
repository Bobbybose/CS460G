# Author: Bobby Bose
# Assignment 2: Linear and Polynomial Regression


# Imports
from functools import partial
import numpy as np
#import math
import pandas as pd

from Assignment1.main import synthetic_data
#pd.options.display.max_rows = 1000
#import matplotlib.pyplot as plt

def main():
# PART 1 of Assignment----------------------------------------------------------------------------------------------

    # Wine data features
    wine_features = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", 
                   "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", 
                   "pH", "sulphates", "alcohol"
                    ]

    # Reading in wine data
    wine_data_df = pd.read_csv("datasets/winequality-red.csv", delimiter = ",")

    # Running the linear regression over the wine data
    wine_weights, wine_MSE = multiple_linear_regression(wine_data_df, wine_features, "quality")

    # Printing the final weights and MSE
    print("Final Wine Weights: " + str(wine_weights))
    print("Final Wine MSE: " + str(wine_MSE))

# PART 2 of Assignment----------------------------------------------------------------------------------------------

    synthetic_data_df_1 = pd.read_csv("datasets/synthetic-1.csv", delimiter = ",", names = ["x", "y"])
    synthetic_data_df_2 = pd.read_csv("datasets/synthetic-2.csv", delimiter = ",", names = ["x", "y"])

    synthetic_dataset_list = [synthetic_data_df_1, synthetic_data_df_2]

    polynomial_values = [2, 3, 5]

    for dataset in synthetic_dataset_list:
        for poly_value in polynomial_values:
            weights, MSE = polynomial_regression(dataset, poly_value)




# main()


# Description: Runs a linear regression over the dataset
# Arguments: Dataset of examples, features of the dataset, class label for the dataset
# Returns: Weights and MSE
def multiple_linear_regression(dataset, features, class_label):
    
    # Randomly initializing the weights
    weights = np.random.uniform(0, 1, len(features) + 1)

    # Parameters
    y_values = dataset[class_label].to_numpy()
    epochs = 750
    alpha = 0.0001
    MSE = 0


    # Adjusting number of times according to epochs values
    for epoch in range(epochs):

        if epoch % 50 == 0:
            print("Epoch: " + str(epoch))
            print("     Weights: " + str(weights))
            print("     MSE: " + str(get_MSE(dataset, weights, y_values)))

        # Updating each weight
        for weight_index in range(weights.size):
            weights[weight_index] -= alpha * total_loss(dataset, weights, y_values, weight_index)

    MSE = get_MSE(dataset, weights, y_values)
    
    return weights, MSE
# multiple_linear_regression()


# Description: Calculates the loss for the regression equation
# Arguments: Dataset of examples, weights for the equation, y values for the equation, index for the weight/feature being updated
# Returns: Loss  
def total_loss(dataset, weights, y_values, weight_index):
    
    loss = 0

    # Summing the loss from each example
    for index, data in dataset.iterrows():
        # Retrieving and cleaning up the x_values for the equation 
        x_values = data.to_numpy()
        x_values = np.delete(x_values, -1)
        x_values = np.insert(x_values, 0, 1)
        
        # Calculating the loss for this example
        loss += (np.sum(np.multiply(weights, x_values)) - y_values[index]) * x_values[weight_index]

    return (1/len(y_values)) * loss
# total_loss()


# Description: Calculates and returns the MSE
# Arguments: Dataset of examples, weights for the equation, y values for the equation
# Returns: MSE
def get_MSE(dataset, weights, y_values):

    loss = 0

    # Summing the loss from each example
    for index, data in dataset.iterrows():
        # Retrieving and cleaning up the x_values for the equation 
        x_values = data.to_numpy()
        x_values = np.delete(x_values, -1)
        x_values = np.insert(x_values, 0, 1)

        # Calculating the loss for this example
        loss += (np.sum(np.multiply(weights, x_values)) - y_values[index]) ** 2

    # Returning the full MSE
    return (1/len(y_values)) * loss 
# get_MSE


def polynomial_regression(dataset, n):
    # Randomly initializing the weights
    weights = np.random.uniform(0, 1, n+1)
    
    # x and y values of the dataset
    x_values = dataset["x"].to_numpy()
    y_values = dataset["y"].to_numpy()

main()