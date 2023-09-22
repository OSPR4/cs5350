import pandas as pd
from decision_tree_node import DecisionTreeNode
from information_gain import information_gain_entropy, information_gain_gini, information_gain_me

class DecisionTree:
    def __init__(self, info_gain_variant='entropy', max_depth=None):
        #todo: add max depth and information gain type
        self.root = None
        # self.data_set = None
        # self.attributes = {}
        # self.labels = None
        self.info_gain_variant = info_gain_variant
        self.max_depth = max_depth
        self.current_depth = 0

    def fit(self, data_set, attributes, labels):
        # Check for numeric attributes
        for attribute in attributes:
            if 'numeric' in attributes[attribute]:
                self.handle_numeric_attribute(data_set, attribute, attributes)

        self.root = self.build_tree_id3(data_set, attributes, labels, curr_depth=0)


    def handle_numeric_attribute(self, data_set, attribute, attributes):
        madian_value = data_set[attribute].median()
        binary_values = [0, 1]
        data_set[attribute] = data_set[attribute].apply(lambda x: binary_values[0] if x < madian_value else binary_values[1])
        attributes[attribute] = binary_values


    def predict(self, data_set, attributes):
        if self.root:
            # Check for numeric attributes
            for attribute in attributes:
                if 'numeric' in attributes[attribute]:
                    self.handle_numeric_attribute(data_set, attribute, attributes)
            
            labels = pd.DataFrame({'label': []})
            for index, row in data_set.iterrows():
                labels.loc[index] = self.predict_row(row, self.root)
            return labels


    def predict_row(self, row, node):
        if node.label:
            return node.label
        else:
            attribute_value = row[node.attribute]
            next_node = node.childern[attribute_value]
            return self.predict_row(row, next_node)


    def get_best_attribute(self, data_set, attributes, labels):
        if self.info_gain_variant == 'mg':
            return information_gain_me(data_set, attributes, labels)
        elif self.info_gain_variant == 'gini':
            return information_gain_gini(data_set, attributes, labels)
        else:
            return information_gain_entropy(data_set, attributes, labels)
    

    # ID3 algorithm
    def build_tree_id3(self, data_set, attributes, labels, curr_depth):
        # self.data_set = data_set
        # self.attributes = attributes
        # self.labels = labels
        self.current_depth = curr_depth
        label_name = data_set.columns[-1]
        if len(set(labels)) == 1: # If all examples have the same label, return a leaf node with that label
            similar_label = labels.iloc[0]
            return DecisionTreeNode(label=similar_label)
        elif len(attributes) == 0:  # If attributes is empty, return a leaf node with the most common label
            most_common_label = labels.value_counts().idxmax()
            return DecisionTreeNode(label=most_common_label)
        else:
            root = DecisionTreeNode()
            best_attribute = self.get_best_attribute(data_set, attributes, labels)
            root.attribute = best_attribute
            for attribute_value in attributes[best_attribute]:
                attribute_value_subset = data_set[data_set[best_attribute] == attribute_value] ## 
                if len(attribute_value_subset) == 0: # If attribute_value_subset is empty, add a leaf node with the most common label in the data_set
                    # most_common_label = self.most_common_label(data_set.label)
                    most_common_label = labels.value_counts().idxmax()
                    root.add_child(attribute_value, DecisionTreeNode(label=most_common_label))
                else:
                    updated_attributes = attributes.copy()
                    del updated_attributes[best_attribute]
                    sub_tree = self.build_tree_id3(attribute_value_subset, updated_attributes, attribute_value_subset[label_name], curr_depth+1)
                    root.add_child(attribute_value, sub_tree)
            return root