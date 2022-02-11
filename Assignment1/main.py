# Author: Bobby Bose
# Assignment 1: Decision Trees


# Imports
from hashlib import new
import pdb
from attr import attr
import numpy as np
import math
import pandas as pd
pd.options.display.max_rows = 1000


# Global variables
MAX_DEPTH = 3


# Description: Each data sample is stored in a Data class
class Data:
    # Holds attributes and values in a dictionary structure
    #   {Attribute_Type: Attribute_Value}
    attribute_values = {}
    # Holds the value for the class label
    class_label_value = ""

    def __str__(self):
        output_string = ""
        for attribute in self.attribute_values:
            output_string += attribute + ": " + self.attribute_values[attribute] + " "

        return output_string


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
    # Child node
    child = Node()
    
    # When initialized, set the parent node, and the branch value of the attribute parent node splits on
    def __init__(self, parent, attribute_value):
        self.parent = parent
        self.attribute_value = attribute_value


def main():

    # Reading in synthetic data
    synthetic_data_1 = pd.read_csv("datasets/synthetic-1.csv", delimiter = ",", names = ["x", "y", "label"])
    #synthetic_data_2 = pd.read_csv("datasets/synthetic-2.csv", delimiter = ",", names = ["x", "y", "label"])
    #synthetic_data_3 = pd.read_csv("datasets/synthetic-3.csv", delimiter = ",", names = ["x", "y", "label"])
    #synthetic_data_4 = pd.read_csv("datasets/synthetic-4.csv", delimiter = ",", names = ["x", "y", "label"])

    #synthetic_data_parts = [synthetic_data_1, synthetic_data_2, synthetic_data_3, synthetic_data_4]
    #synthetic_data_full = pd.concat(synthetic_data_parts)

    dataset = parse_data(synthetic_data_1)

    for data in dataset: 
        print(dataset)
    #ID3()    
    

def parse_data(dataframe_dataset):
    
    dataset = np.array([])

    for index, row in dataframe_dataset.iterrows():

        new_data = Data()

        for column in dataframe_dataset.columns:
            Data.attribute_values[column] = row.loc[column]

        dataset = np.append(dataset, new_data)

    return dataset


# Description: Main decision tree creation function.
# Arguments: dataset (examples), class label for the dataset, array of attribute objects
# Returns: Root of the current tree (subtree starting at root)
def ID3(dataset, class_label, attributes):
    
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
        splitting_attribute = best_attribute(attributes)
        root.attribute = splitting_attribute

        for attribute_value in attributes[splitting_attribute]:
            new_branch = Branch(root, attribute_value)
            subset = split_dataset(dataset, splitting_attribute, attribute_value)

            if subset.size == 0:
                new_node = Node()
                new_node.attribute = most_common_class_label_value(dataset, class_label)
            else:
                attributes = attributes[attributes != root.attribute]
                ID3(subset, class_label, attributes)    
    
    return root


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


# Description: Calculate information gain for splitting on an attribute
# Arguments: dataset (examples), class label for the dataset, attribute splitting on
# Returns: Information gain value for given attribute
def information_gain(dataset, class_label, chosen_attribute):
  
    # Track number of occurrences of each value of the chosen attribute in the dataset
    attribute_value_count = {}

    for value in chosen_attribute.values:
        attribute_value_count[value] = 0

    for data in dataset:
        attribute_value_count[data.attribute_values[chosen_attribute]] += 1

    #Calculating Entropy
    average_child_entropy = 0

    for value in attribute_value_count:
        # Obtaining subset of dataset split on chosen attribute
        new_dataset = split_dataset(dataset, chosen_attribute, value)
        average_child_entropy += (attribute_value_count[value]/dataset) * entropy(new_dataset, class_label)

    return entropy(dataset, class_label) - average_child_entropy


# Description: Splits a dataset based on the value of a given attribute
# Arguments: dataset (examples), attribute splitting on, attribute value wanted
# Returns: New dataset split on given attribute
def split_dataset(dataset, chosen_attribute, chosen_attribute_value):
    
    new_dataset = np.array([])
    
    #Spliting dataset based on given attribute and storing data with chosen_attribute_value
    for data in dataset:
        if data.attribute_values[chosen_attribute.attribute_name] == chosen_attribute_value:
            new_dataset.append(data)

    return new_dataset
# split_dataset()


# Description: Calculates entropy for a dataset
# Arguments: dataset (examples), class label for the dataset
# Returns: Entropy of dataset
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