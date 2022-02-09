# Author: Bobby Bose
# Assignment 1: Decision Trees


# Imports
import numpy as np
import math


# Global variables
MAX_DEPTH = 3


# Description: Each data sample is stored in a Data class
class Data:
    # Holds attributes and values in a dictionary structure
    #   {Attribute_Type: Attribute_Value}
    attribute_values = {}

    # Holds the value for the class label
    class_label_value = ""


# Description: Each attribute is stored in a Attribute class
class Attribute:
    # List of possible values
    values = np.array([])

    # Initialize attribute and store name (attribute label)
    def __init__(self, attribute_name):
        self.attribute_name = attribute_name


# Description: The class_label (Target Attribute) for a dataset
class Class_Label:
    # Set the positive and negative label values at initialization
    def __init__(self, positive_label, negative_label):
        self.positive_label = positive_label
        self.negative_label = negative_label


# Description: Nodes (leaves) of the tree are stored in Node class
class Node:
    # Attribute splitting on
    attribute = ""
    # Array of Branch objects that connect this Node to child nodes
    branches = np.array([])
        

# Description: Branches connect parent and child nodes together
class Branch:
    # When initialized, set the parent node, and the branch value of the feature parent node splits on
    def __init__(self, parent, feature_value):
        self.parent = parent
        self.feature_value = feature_value

    # Set child node. Can not happen till after initialization in decision tree creation
    def add_child(self, child):
        self.child = child


def main():
    create_decision_tree()    
    

# Description: Main decision tree creation function. Equivalent to ID3
# Arguments: dataset (examples), class label for the dataset, array of attribute objects
# Returns: Root of the current tree (subtree starting at root)
def create_decision_tree(dataset, class_label, attributes):
    
    # Root node of the tree beginning here
    root = Node()

    # Checking if there is only one type of class label left
    #   If so, set it to the root's attribute and return
    if num_unique_labels_in_dataset(dataset) == 1:
        root.attribute = dataset[0].class_label_value
        return root
    
    # Checking if there are no more attributes left
    #   If so, return the most common class label value
    if attributes.size == 0:
        root.attribute = most_common_class_label_value(dataset, class_label)
        return root
    else:
        root.attribute = best_attribute(attributes)
        
    
# Description: Checks if there is only one class label value left in the dataset
# Arguments: dataset (examples)
# Returns: Number of unique class labels in the dataset
def num_unique_labels_in_dataset(dataset):
    
    # Store number of unique labels (Theoretically only 1 or 2)
    num_unique_labels = 0
    # Stores the different label values left 
    labels = np.array([])

    # Cycling through all the data and storing unique labels
    for data in dataset:
        if not data.class_label_value in labels:
            num_unique_labels += 1
            labels.append(data.class_label_value)
    
    return num_unique_labels
# num_unique_labels_in_dataset


# Description: Finds and returns the most common class label value in the dataset
# Arguments: dataset (examples), class label for the dataset
# Returns: Most common class label value in the dataset
def most_common_class_label_value(dataset, class_label):

    # Keep track of number of occurrences of each class_label value    
    label_value_count = {"positive": 0, "negative": 0}
    # Keeps track of highest occurrence value count
    maxCount = 0
    # Keeps track of highest occurrence value
    most_common_value = ""


    # Tally up occurrences for each class_label value
    for data in dataset:
        # Adding an occurrence to the current data's class_label_value
        label_value_count["positive"], label_value_count["negative"] = class_label_occurrences(dataset, class_label)

        # Updating the other tracking variables if needed
        if label_value_count["positive"] > maxCount:
            maxCount = label_value_count["positive"]
            most_common_value = "positive"
        else:
            maxCount = label_value_count["negative"]
            most_common_value = "negative"
    
    # Returning most common class label
    return most_common_value
# most_common_label_value


# Description: Calculates how many positive and negative class labels are in the dataset
# Arguments: dataset (examples), class label for the dataset
# Returns: Number of positive and negative class label occurrences in the dataset
def class_label_occurrences(dataset, class_label):
    # Stores numbers of positive/negative class label occurrences
    num_positive = 0
    num_negative = 0
  
    # Tally up occurrences for each class_label value
    for data in dataset:
        # Adding an occurrence to the current data's class_label_value
        if data.class_label_value == class_label.positive_label:
            num_positive += 1
        else:
            num_negative += 1

    return num_positive, num_negative
# class_label_occurrences


# Description: Selects the best attribute to split on at this position in the tree
# Arguments: dataset (examples), class label for the dataset, array of attribute objects
# Returns: The best attribute to split on
def best_attribute(dataset, class_label, attributes):

    best_attribute = ""
    best_info_gain = 0

    # Cycling through all the attributes and calculating the information gain
    for attribute in attributes:
        curr_info_gain = information_gain(dataset, class_label, attribute)

        # If the information gain 
        if curr_info_gain > best_info_gain:
            best_info_gain = curr_info_gain
            best_attribute = attribute

    return best_attribute
# best_attribute()


# Description: 
# Arguments: dataset (examples), class label for the dataset
# Returns:
def information_gain(dataset, class_label, chosen_attribute):
    attribute_value_count = {}

    for value in chosen_attribute.values:
        attribute_value_count[value] = 0

    for data in dataset:
        attribute_value_count[data.attribute_values[chosen_attribute]] += 1

    average_child_entropy = 0

    for value in attribute_value_count:
        new_dataset = split_dataset(dataset, chosen_attribute, value)
        
        average_child_entropy += (attribute_value_count[value]/dataset) * entropy(new_dataset, class_label)

    return entropy(dataset, class_label) - average_child_entropy


# Description: 
# Arguments: 
# Returns:
def split_dataset(dataset, chosen_attribute, chosen_attribute_value):
    
    new_dataset = np.array([])
    
    for data in dataset:
        if data.attribute_values[chosen_attribute.attribute_name] == chosen_attribute_value:
            new_dataset.append(data)

    return new_dataset
# split_dataset()


# Description: 
# Arguments: dataset (examples), class label for the dataset
# Returns:
def entropy(dataset, class_label):   
    # Retrieving class_label split
    num_positive, num_negative = class_label_occurrences(dataset, class_label)

    if num_positive == 0 or num_negative == 0:
        return 0

    # Calculating positive and negative class_label parts of entropy calculation
    positive = (-num_positive/dataset.size) * (math.log(num_positive/dataset.size , 2))
    negative = (-num_negative/dataset.size) * (math.log(num_negative/dataset.size , 2))

    # Returning positive part - negative part
    return positive - negative
# entropy()


main()