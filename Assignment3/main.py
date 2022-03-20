# Author: Bobby Bose
# Assignment 3: Multilayer Perceptrons 


# Imports
import math
import numpy as np
import pandas as pd


# Main function of the program and neural net
def main():

    # Reading in training and testing datasets
    training_dataset_df = pd.read_csv("datasets/mnist_train_0_1.csv")
    test_dataset_df = pd.read_csv("datasets/mnist_test_0_1.csv")
    
    training_data_x = list(map(lambda x: x/255, training_dataset_df.iloc[:, 1:].to_numpy()))
    training_data_y = training_dataset_df.iloc[:, 0].tolist()
    test_data_x = list(map(lambda x: x/255, test_dataset_df.iloc[:, 1:].to_numpy()))
    test_data_y = test_dataset_df.iloc[:, 0].tolist()

    # 784 Input features
    # Structure: 784 -> 50 -> 1
    num_nodes_in_layer = [784, 50, 1]

    # 784x50
    weights_h = []
    # 50x1
    weights_o = []

    for i in range(num_nodes_in_layer[0]):
        weights_h.append(np.random.uniform(-1, 1, num_nodes_in_layer[1]).tolist())

    for i in range(num_nodes_in_layer[1]):
        weights_o.append(np.random.uniform(-1, 1, num_nodes_in_layer[2]).tolist())

    # 50x1
    bias_h = np.random.uniform(0, 1, num_nodes_in_layer[1]).tolist()
    # 1x1
    bias_o = np.random.uniform(0, 1, num_nodes_in_layer[2]).tolist()

    print(bias_h)
    print(bias_o)

    alpha = 0.5
    epochs = 1

    for epoch in range(epochs):
        for index, data in enumerate(training_data_x):
            # data is [784 x 1]
            forward_pass(weights_h, weights_o, bias_h, bias_o, data)




def forward_pass(weights_h, weights_o, bias_h, bias_o, x):
    # 50x1
    in_h = np.dot(np.array(weights_h).transpose(), x) + bias_h


# Description: Calculates sigmoid function result of x
# Arguments: x
# Returns: Sigmoid function of x
def sigmoid(x):
    return 1/(1+math.exp(-x))


# Description: Calculates derivative of sigmoid function result of x
# Arguments: x
# Returns: Derivative of sigmoid of x
def sigmoid_prime(x):
    return sigmoid(x) * (1-sigmoid(x))


main()