# Author: Bobby Bose
# Assignment 1: Decision Trees


# Imports
from curses import raw
from hashlib import new
import pdb
from re import A
from attr import attr
#from matplotlib.pyplot import cla
import numpy as np
import math
import pandas as pd
pd.options.display.max_rows = 1000


# Global variables
MAX_DEPTH = 3
SYNTHETIC_CLASS_LABEL = "class"
BINS = 4

# Description: Each data sample is stored in a Data class
class Data:
    # Holds attributes and values in a dictionary structure
    #   {Attribute_Type: Attribute_Value}
    attribute_values = {}
    
    # Class label
    class_label = ""

    # Holds the value for the class label
    class_label_value = ""

    def __init__(self, attribute_values = {}, class_label = "", class_label_value = ""):
        self.attribute_values = attribute_values
        self.class_label = class_label
        self.class_label_value = class_label_value

    def __str__(self):
        output_string = ""
        for attribute in self.attribute_values:
            output_string += attribute + ": " + str(self.attribute_values[attribute]) + " "
        output_string += self.class_label + ": " + str(self.class_label_value)
        return output_string


# Description: Each attribute is stored in a Attribute class
class Attribute:
    # List of possible values
    values = []

    # Initialize attribute and store name (attribute label)
    def __init__(self, attribute_name = ""):
        self.attribute_name = attribute_name


# Description: Nodes (leaves) of the tree are stored in Node class
class Node:
    # Attribute splitting on
    attribute = Attribute()

    # Hold children nodes
    child_nodes = []

    # Array of Branch objects that connect this Node to child nodes
    branches = []

    def __init__(self, value = "Root"):
        self.value = value
        

# Description: Branches connect parent and child nodes together
class Branch:
    # Child node
    child = Node()
    
    # When initialized, set the parent node, and the branch value of the attribute parent node splits on
    def __init__(self, parent, attribute_value):
        self.parent = parent
        self.attribute_value = attribute_value


def main():
    synthetic_data()
# main()


# Description: Train and test the Decision Trees for the synthetic data
# Arguments: None
# Returns: 
def synthetic_data():

    # Reading in synthetic data
    synthetic_data_df_1 = pd.read_csv("datasets/synthetic-1.csv", delimiter = ",", names = ["x", "y", "class"])
    synthetic_data_df_2 = pd.read_csv("datasets/synthetic-2.csv", delimiter = ",", names = ["x", "y", "class"])
    synthetic_data_df_3 = pd.read_csv("datasets/synthetic-3.csv", delimiter = ",", names = ["x", "y", "class"])
    synthetic_data_df_4 = pd.read_csv("datasets/synthetic-4.csv", delimiter = ",", names = ["x", "y", "class"])

    # Collect all the raw DataFrame data into one list
    raw_synthetic_dataset_list = [synthetic_data_df_1, synthetic_data_df_2, synthetic_data_df_3, synthetic_data_df_4]
    
    # Copying data to preserve original for testing
    synthetic_dataset_list = raw_synthetic_dataset_list.copy()
    
    for dataset in synthetic_dataset_list:
        dataset["x"] = pd.qcut(dataset["x"], BINS)
        dataset["y"] = pd.qcut(dataset["y"], BINS)

    # Attributes in the synthetic data
    #synthetic_data_attributes = [Attribute("x"), Attribute("y")]
    

    # Process the raw synthetic data and obtain easier-to-work-with datasets
    #synthetic_data = data_processing(synthetic_data_list, synthetic_data_attributes, SYNTHETIC_CLASS_LABEL)

    synthetic_data_trees = []

    #for data in synthetic_data[0]:
    #    print(data)

    synthetic_class_label = "class"
    synthetic_attributes = ["x", "y"]
    synthetic_data_trees.append(Decision_Tree(synthetic_dataset_list[0], synthetic_class_label, synthetic_attributes))

    for tree in synthetic_data_trees:
        print(tree)

    #return 1

    #for datalist in synthetic_data: 
    #    synthetic_data_trees.append(Decision_Tree(datalist, SYNTHETIC_CLASS_LABEL, synthetic_data_attributes))

# synthetic_data()


# Description: Process raw data to an easier format to work with 
# Arguments: Array of raw DataFrame data, attributes for this selection of data, class label for the dataset
# Returns: Processed datasets
def data_processing(raw_data_list, data_attributes, class_label):
    
    # To collect process datasets
    datasets = []

    # Discretizing the data
    for datalist in raw_data_list:
        for attribute in data_attributes:
            datalist[attribute.attribute_name] = pd.qcut(datalist[attribute.attribute_name], BINS)#, labels = False)      
            attribute.values = datalist[attribute.attribute_name].unique()

        # Converting the dataset to a dict for easier parsing
        dataset_dict = datalist.to_dict(orient = 'index')

        # Parse and collect dataset information
        datasets.append(parse_data(dataset_dict, class_label))

    return datasets
# data_processing()


# Description: Parses dataset dict data into an array of Data objects
# Arguments: Dictionary of raw data, class label for the dataset
# Returns: Array of Data objects
def parse_data(dataset_dict, class_label):
    
    # Will be an array of Data objects
    dataset = []

    # Populating dataset with Data objects from the raw data
    for index, data in dataset_dict.items():
        class_label_value = data.pop(class_label)
        dataset.append(Data(data, class_label, class_label_value))

    return dataset
# parse_data()


class Decision_Tree:

    # Root node of the entire tree
    root_node = Node()


    # Description: Decision tree initialization. Creates tree and sets root node
    def __init__(self, dataset, class_label, attributes):
        self.root_node = self.ID3(dataset, class_label, attributes)
        

    # Description: Main decision tree creation function.
    # Arguments: dataset (examples), class label for the dataset, array of attribute objects
    # Returns: Root of the current tree (subtree starting at root)
    def ID3(self, dataset, class_label, attributes):

        root = Node()

        # Checking if there is only one type of class label left
        if len(dataset[class_label].unique()) == 1:
            root.attribute = dataset.loc[0][class_label]
            return root
        
        # Checking if there are no more attributes left
        if len(attributes) == 0:
            root.attribute = dataset[class_label].mode()
            return root
              
        # Finding best attribute to split on
        splitting_attribute = self.best_attribute(dataset, class_label, attributes)
        
        # Setting root attribute
        root.attribute = splitting_attribute

        # Cycling through attribute values are creating branches
        for attribute_value in dataset[splitting_attribute].values.unique():
            
            # Creating subset of dataset with current attribute_value
            subset = dataset.loc[dataset[splitting_attribute] == attribute_value]

            if len(subset) == 0:
                new_node = Node(attribute_value)
                new_node.attribute = self.most_common_class_label_value(dataset, class_label)
                root.child_nodes.append(new_node)

            else:
                new_attributes = []
                for attribute in attributes:
                    if attribute != root.attribute:
                        new_attributes.append(attribute)
                
                new_node = self.ID3(subset, class_label, new_attributes)
                new_node.value = attribute_value
                root.child_nodes.append(new_node)
    
        print("main root: " + root.attribute.attribute_name)
        return root
    # ID3()


    # Description: Checks if there is only one class label value left in the dataset
    # Arguments: dataset (examples)
    # Returns: Number of unique class labels in the dataset
    def num_unique_labels_in_dataset(self, dataset):
        
        # Store number of unique labels (Theoretically only 1 or 2)
        num_unique_labels = 0
        # Stores the different label values left 
        labels = []

        # Cycling through all the data and storing unique labels
        for data in dataset:
            if not data.class_label_value in labels:
                num_unique_labels += 1
                labels.append(data.class_label_value)
        
        return num_unique_labels
    # num_unique_labels_in_dataset()


    # Description: Finds and returns the most common class label value in the dataset
    # Arguments: dataset (examples), class label for the dataset
    # Returns: Most common class label value in the dataset
    def most_common_class_label_value(self, dataset, class_label):

        # Keep track of number of occurrences of each class_label value    
        label_value_count = {"positive": 0, "negative": 0}
        # Keeps track of highest occurrence value count
        maxCount = 0
        # Keeps track of highest occurrence value
        most_common_value = ""


        # Tally up occurrences for each class_label value
        for data in dataset:
            # Adding an occurrence to the current data's class_label_value
            label_value_count["positive"], label_value_count["negative"] = self.class_label_occurrences(dataset, class_label)

            # Updating the other tracking variables if needed
            if label_value_count["positive"] > maxCount:
                maxCount = label_value_count["positive"]
                most_common_value = "positive"
            else:
                maxCount = label_value_count["negative"]
                most_common_value = "negative"
        
        # Returning most common class label
        return most_common_value
    # most_common_label_value()


    # Description: Calculates how many positive and negative class labels are in the dataset
    # Arguments: dataset (examples), class label for the dataset
    # Returns: Number of positive and negative class label occurrences in the dataset
    def class_label_occurrences(self, dataset, class_label):
    
        # Stores numbers of positive/negative class label occurrences
        num_positive = 0
        num_negative = 0
    
        # Tally up occurrences for each class_label value
        for data in dataset:
            # Adding an occurrence to the current data's class_label_value
            if data.class_label_value == 1:
                num_positive += 1
            else:
                num_negative += 1

        return num_positive, num_negative
    # class_label_occurrences()


    # Description: Selects the best attribute to split on at this position in the tree
    # Arguments: dataset (examples), class label for the dataset, array of attribute objects
    # Returns: The best attribute to split on
    def best_attribute(self, dataset, class_label, attributes):

        best_attribute = ""
        best_info_gain = 0

        # Cycling through all the attributes and calculating the information gain
        for attribute in attributes:
            curr_info_gain = self.information_gain(dataset, class_label, attribute)

            # If the information gain 
            if curr_info_gain > best_info_gain:
                best_info_gain = curr_info_gain
                best_attribute = attribute

        return best_attribute
    # best_attribute()


    # Description: Calculate information gain for splitting on an attribute
    # Arguments: dataset (examples), class label for the dataset, attribute splitting on
    # Returns: Information gain value for given attribute
    def information_gain(self, dataset, class_label, chosen_attribute):
    
        # Track number of occurrences of each value of the chosen attribute in the dataset
        attribute_value_count = {}

        # Filling in dict
        for attribute_value in dataset[chosen_attribute].values.unique():
            attribute_value_count[attribute_value] = 0

        # Tallying occurrences
        for index, data in dataset.iterrows():
            attribute_value_count[data[chosen_attribute]] += 1

        #Calculating Entropy
        average_child_entropy = 0

        for value in attribute_value_count:
            # Obtaining subset of dataset split on chosen attribute
            new_dataset = dataset.loc[dataset[chosen_attribute] == value]

            average_child_entropy += (attribute_value_count[value]/len(dataset)) * self.entropy(new_dataset, class_label)

        return self.entropy(dataset, class_label) - average_child_entropy
    # information_gain()


    # Description: Splits a dataset based on the value of a given attribute
    # Arguments: dataset (examples), attribute splitting on, attribute value wanted
    # Returns: New dataset split on given attribute
    def split_dataset(self, dataset, chosen_attribute, chosen_attribute_value):
        
        new_dataset = []
        
        #Spliting dataset based on given attribute and storing data with chosen_attribute_value
        for data in dataset:
            if data.attribute_values[chosen_attribute.attribute_name] == chosen_attribute_value:
                new_dataset.append(data)

        return new_dataset
    # split_dataset()


    # Description: Calculates entropy for a dataset
    # Arguments: dataset (examples), class label for the dataset
    # Returns: Entropy of dataset
    def entropy(self, dataset, class_label):   
        
        # Stores numbers of positive/negative class label occurrences
        num_positive = 0
        num_negative = 0
        
       # Tally up occurrences for each class_label value
        for index, data in dataset.iterrows():
            # Adding an occurrence to the current data's class_label_value
            if data[class_label] == 1:
                num_positive += 1
            else:
                num_negative += 1 

        # Case when all labels are the same
        if num_positive == 0 or num_negative == 0:
            return 0
        
        # Calculating positive and negative class_label parts of entropy calculation
        positive = (-num_positive/len(dataset)) * (math.log(num_positive/len(dataset) , 2))
        negative = (-num_negative/len(dataset)) * (math.log(num_negative/len(dataset) , 2))

        # Returning positive part - negative part
        return positive + negative
    # entropy()


    def __str__(self):
        self.print_tree(self.root_node, 0)
        return ""


    def print_tree(self, root, level):
        
        for i in range(level):
            print("--", end='')

        if len(root.child_nodes) == 0:
            print(root.attribute)

        for child in root.child_nodes:
            print(root.attribute.attribute_name + ": " + str(child.value))
            self.print_tree(child, level+1)
            

main()