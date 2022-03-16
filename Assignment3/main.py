# Author: Bobby Bose
# Assignment 3: Multilayer Perceptrons 


# Imports
import numpy as np
import pandas as pd


def main():


    training_dataset_df = pd.read_csv("datasets/mnist_train_0_1.csv", delimiter = ",")
    testing_dataset_df = pd.read_csv("datasets/mnist_test_0_1.csv", delimiter = ",")

    training_data = training_dataset_df.to_numpy()
    test_data = testing_dataset_df.to_numpy()

    # 784 Input features
    # Structure: 784 -> 50 -> 1
    num_nodes_in_layer = [784, 50, 1]

    weights_h = []
    weights_o = []

    for i in range(num_nodes_in_layer[0]):
        weights_h.append(np.random.uniform(-1, 1, num_nodes_in_layer[1]).tolist())

    for i in range(num_nodes_in_layer[1]):
        weights_o.append(np.random.uniform(-1, 1, num_nodes_in_layer[2]).tolist())

    bias_h = np.random.uniform(0, 1, num_nodes_in_layer[1]).tolist()
    bias_o = np.random.uniform(0, 1, num_nodes_in_layer[2]).tolist()

    print(bias_h)
    print(bias_o)


main()