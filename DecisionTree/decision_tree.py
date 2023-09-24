import pandas as pd
import copy
from decision_tree_node import DecisionTreeNode
from information_gain import information_gain_entropy, information_gain_gini, information_gain_me

class DecisionTree:
    def __init__(self, info_gain_variant='entropy', max_depth=None):
        self.root = None
        self.info_gain_variant = info_gain_variant
        self.max_depth = max_depth
        self.current_depth = 0

    def fit(self, data_set, attributes, labels):
        dt_attributes = copy.deepcopy(attributes)
        dt_data_set = copy.deepcopy(data_set)
        for attribute in dt_attributes:
            if 'numeric' in dt_attributes[attribute]:
                self.handle_numeric_attribute(dt_data_set, attribute, dt_attributes)

        self.root = self.build_tree_id3(dt_data_set, dt_attributes, labels, curr_depth=0)


    def handle_numeric_attribute(self, data_set, attribute, attributes):
        madian_value = data_set[attribute].median()
        binary_values = [0, 1]
        data_set[attribute] = data_set[attribute].apply(lambda x: binary_values[0] if x < madian_value else binary_values[1])
        attributes[attribute] = binary_values


    def predict(self, data_set, attributes):
        if self.root:
            dt_attributes = copy.deepcopy(attributes)
            dt_data_set = copy.deepcopy(data_set)
            for attribute in dt_attributes:
                if 'numeric' in dt_attributes[attribute]:
                    self.handle_numeric_attribute(dt_data_set, attribute, dt_attributes)
            
            predicted_labels = pd.DataFrame({'label': []})
            for index, row in dt_data_set.iterrows():
                predicted_labels.loc[index] = self.predict_row(row, self.root)
            return predicted_labels


    def predict_row(self, row, node):
        if node.label:
            return node.label
        else:
            attribute_value = row[node.attribute]
            next_node = node.childern[attribute_value]
            return self.predict_row(row, next_node)


    def get_best_attribute(self, data_set, attributes, labels):
        if self.info_gain_variant == 'me':
            return information_gain_me(data_set, attributes, labels)
        elif self.info_gain_variant == 'gini':
            return information_gain_gini(data_set, attributes, labels)
        else:
            return information_gain_entropy(data_set, attributes, labels)
    

    # ID3 algorithm
    def build_tree_id3(self, data_set, attributes, labels, curr_depth):
        self.current_depth = curr_depth
        label_name = data_set.columns[-1]
        if len(set(labels)) == 1: 
            similar_label = labels.iloc[0]
            return DecisionTreeNode(label=similar_label)
        elif len(attributes) == 0:  
            most_common_label = labels.value_counts().idxmax()
            return DecisionTreeNode(label=most_common_label)
        else:
            root = DecisionTreeNode()
            best_attribute = self.get_best_attribute(data_set, attributes, labels)
            root.attribute = best_attribute

            if curr_depth + 1 == self.max_depth:
                for attribute_value in attributes[best_attribute]:
                    attribute_value_subset = data_set[data_set[best_attribute] == attribute_value]
                    if len(attribute_value_subset) == 0:
                        most_common_label = labels.value_counts().idxmax()
                        root.add_child(attribute_value, DecisionTreeNode(label=most_common_label))
                    else:
                        most_freq_label = attribute_value_subset[label_name].value_counts().idxmax()
                        root.add_child(attribute_value, DecisionTreeNode(label=most_freq_label))
            else:
                for attribute_value in attributes[best_attribute]:
                    attribute_value_subset = data_set[data_set[best_attribute] == attribute_value] ## 
                    if len(attribute_value_subset) == 0: 
                        most_common_label = labels.value_counts().idxmax()
                        root.add_child(attribute_value, DecisionTreeNode(label=most_common_label))
                    else:
                        updated_attributes = attributes.copy()
                        del updated_attributes[best_attribute]
                        sub_tree = self.build_tree_id3(attribute_value_subset, updated_attributes, attribute_value_subset[label_name], curr_depth+1)
                        root.add_child(attribute_value, sub_tree)
            return root