# Author: Bobby Bose
# Assignment 2: Linear and Polynomial Regression


# Imports
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
    
    wine_attributes = wine_labels.copy().remove("quality")

    wine_data_df = pd.read_csv("datasets/winequality-red.csv", delimiter = ",", names = wine_labels)


    # Preprocessing input features
    wine_data_processed = wine_data_df.copy()
    for attribute in wine_attributes:
        max = wine_data_processed[attribute].max()
        min = wine_data_processed[attribute].min()

        for index, row in wine_data_processed.iterrows():
            row[attribute] = (row[attribute] - min) / (max - min)

    weights, MSE = multiple_linear_regression(wine_data_processed, wine_attributes)

    return 1

# main()

def multiple_linear_regression(dataset, attributes):
    weights = []
    MSE = 0

    weights = np.random.uniform(0, 20, len(attributes)).tolist()

    #for i in range(len(attributes)):
    #    weights.append()

    for attribute in attributes:
        x_values = dataset[attribute].tolist()

