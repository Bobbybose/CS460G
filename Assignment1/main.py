# Author: Bobby Bose
# Assignment 1: Decision Trees

import numpy as np

MAX_DEPTH = 3

class Node:
    
    attribute = ""
    branches = np.array([])
        
class Branch:

    def __init__(self, parent, feature_value):
        self.parent = parent
        self.feature_value = feature_value

    def add_child(self, child):
        self.child = child

def main():
    create_decision_tree()    
    

def create_decision_tree(dataset, class_label, attributes):
    
    root = Node()

    if num_labels_in_dataset(dataset) == 1:
        root.attribute = dataset[0].class_label
        return root
    
    if attributes.size == 0:
        root.attribute = most_common_class_label_value(dataset)
        return root


def num_labels_in_dataset(dataset):
    return 1

def most_common_class_label_value(dataset):
    return 1

main()