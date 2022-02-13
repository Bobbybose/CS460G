# Author: Bobby Bose
# Assignment 1: Decision Trees


# Imports
import matplotlib.pylot as plt
import numpy as np
import math
import pandas as pd
pd.options.display.max_rows = 1000


# Global variables
MAX_DEPTH = 3
NUM_BINS = 4
SYNTHETIC_CLASS_LABEL = "class"
POKEMON_CLASS_LABEL = "Legendary"


# Description: Nodes (leaves) of the tree are stored in Node class
class Node:
    
    def __init__(self, value = "Root"):
        self.attribute = ""
        self.value = value
        self.child_nodes = []
    
    def __str__(self):
        return "Splitting Attribute: " + str(self.attribute) + "    Value: " + str(self.value)

def main():   

    # Classifying synthetic data and visualizing classifiers
    synthetic_data()

    print("\n-----------------------------------------------------------------------------------\n")

    # Classifying pokemon data
    pokemon_data()
# main()


# Description: Train and test the Decision Trees for the synthetic data
# Arguments: None
# Returns: None
def synthetic_data():

    # Reading in synthetic data
    synthetic_data_df_1 = pd.read_csv("datasets/synthetic-1.csv", delimiter = ",", names = ["x", "y", "class"])
    synthetic_data_df_2 = pd.read_csv("datasets/synthetic-2.csv", delimiter = ",", names = ["x", "y", "class"])
    synthetic_data_df_3 = pd.read_csv("datasets/synthetic-3.csv", delimiter = ",", names = ["x", "y", "class"])
    synthetic_data_df_4 = pd.read_csv("datasets/synthetic-4.csv", delimiter = ",", names = ["x", "y", "class"])

    
    # Copying data to preserve original for testing
    synthetic_dataset_list = [synthetic_data_df_1, synthetic_data_df_2, synthetic_data_df_3, synthetic_data_df_4]
    
    # Discretizing the synthetic dataset
    for dataset in synthetic_dataset_list:
        dataset["x"] = pd.qcut(dataset["x"], NUM_BINS)
        dataset["y"] = pd.qcut(dataset["y"], NUM_BINS)

    # Parameters for the synthetic data
    synthetic_dataset_trees = []
    synthetic_class_label = "class"
    synthetic_attributes = ["x", "y"]

    # Training, printing, and on the synthetic data
    for i in range(len(synthetic_dataset_list)):
        synthetic_dataset_trees.append(Decision_Tree(synthetic_dataset_list[i], synthetic_class_label, synthetic_attributes))

        print("Printing Synthetic Tree " + str(i) + ":")
        print(synthetic_dataset_trees[i])
    
    print("\n")
    
    # Testing on synthetic data and printing accuracies
    for i in range(len(synthetic_dataset_trees)):
        accuracy = synthetic_dataset_trees[i].test_on_tree(synthetic_dataset_list[i], SYNTHETIC_CLASS_LABEL)

        print("Synthetic Test Accuracy: " + str(accuracy))

    return synthetic_dataset_list, synthetic_dataset_trees, 

# synthetic_data()


# Description: Train and test the Decision Tree for the pokemon data
# Arguments: None
# Returns: None
def pokemon_data():

    # Reading in pokemon data
    pokemon_stats = pd.read_csv("datasets/pokemonStats.csv", delimiter = ",")
    pokemon_legendary = pd.read_csv("datasets/pokemonLegendary.csv", delimiter = ",")

    # Combining the two DataFrames
    pokemon_dataset = pd.concat([pokemon_stats, pokemon_legendary], axis = 1)

    # Obtaining the attributes
    pokemon_attributes = pokemon_dataset.columns.tolist()
    pokemon_attributes.remove(POKEMON_CLASS_LABEL)

    # Attributes that need to be discretized
    continuous_attributes = ["Total", "HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]

    # Discretizing the pokemon dataset
    for attribute in continuous_attributes:
        pokemon_dataset[attribute] = pd.qcut(pokemon_dataset[attribute], NUM_BINS)

    # Training on the pokemon data
    pokemon_tree = Decision_Tree(pokemon_dataset, POKEMON_CLASS_LABEL, pokemon_attributes)

    # Printing the pokemon tree
    print("Printing Pokemon Tree:")
    print(pokemon_tree)
    print("\n")
    
    # Testing on pokemon data and printing accuracies
    accuracy = pokemon_tree.test_on_tree(pokemon_dataset, POKEMON_CLASS_LABEL)
    print("Synthetic Test Accuracy: " + str(accuracy))

# pokemon_data()


class Decision_Tree:

    # Description: Decision tree initialization. Creates tree and sets root node
    def __init__(self, dataset, class_label, attributes):
        self.root_node = self.ID3(dataset, class_label, attributes, 0)
        

    # Description: Main decision tree creation function.
    # Arguments: dataset (examples), class label for the dataset, array of attribute objects
    # Returns: Root of the current tree (subtree starting at root)
    def ID3(self, dataset, class_label, attributes, depth):

        root = Node()

        # Checking if there is only one type of class label left
        if len(dataset[class_label].unique()) == 1 or len(attributes) == 0 or depth == 3:
            root.attribute = dataset[class_label].value_counts().idxmax()
            return root
              
        # Finding best attribute to split on
        splitting_attribute = self.best_attribute(dataset, class_label, attributes)
        
        # Setting root attribute
        root.attribute = splitting_attribute

        # Getting unique values
        if type(dataset[splitting_attribute].values) == 'pandas.core.arrays.categorical.Categorical':
            unique_values = dataset[splitting_attribute].values.unique()
        else:
            unique_values = np.unique(dataset[splitting_attribute].values)

        # Cycling through attribute values are creating branches
        for attribute_value in unique_values:
            
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
                
                new_node = self.ID3(subset, class_label, new_attributes, depth+1)
                new_node.value = attribute_value
                root.child_nodes.append(new_node)
    
        return root
    # ID3()


    # Description: Testing a dataset on the tree
    # Arguments: dataset being tested
    # Returns: Accuracy of the test
    def test_on_tree(self, test_data, class_label):
        num_correct_predicts = 0

        for index, data in test_data.iterrows():
            if data[class_label] == self.predict_label(data, self.root_node):
                num_correct_predicts += 1

        return num_correct_predicts/len(test_data)

    # test_on_tree()


    # Description: Predict the class label using the decision tree
    # Arguments: Data being tested on
    # Returns: Predicted label
    def predict_label(self, data, root):
        
        # If this node is a leaf, return the label
        if not root.child_nodes:
            return root.attribute

        # Getting this data's attribute value for this node's split
        data_attribute = data[root.attribute]
        
        # Going deeper into the tree
        for child in root.child_nodes:
            if child.value == data[root.attribute]:
                return self.predict_label(data, child)                

    # predict_label()


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
        if type(dataset[chosen_attribute].values) == 'pandas.core.arrays.categorical.Categorical':
            for attribute_value in dataset[chosen_attribute].values.unique():
                attribute_value_count[attribute_value] = 0
        else:
            for attribute_value in np.unique(dataset[chosen_attribute].values):
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


    # Overload the print function
    def __str__(self):
        self.print_tree(self.root_node, 0)
        return ""

    # Print tree in nice format
    def print_tree(self, root, level):
        
        for i in range(level):
            print("   ", end='')

        print(str(root))
        
        for child in root.child_nodes:
            self.print_tree(child, level+1)

main()