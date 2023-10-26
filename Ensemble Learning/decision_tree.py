import pandas as pd
import numpy as np
import random
import copy
from decision_tree_node import DecisionTreeNode
from information_gain import information_gain_entropy, information_gain_gini, information_gain_me

class DecisionTree:
    def __init__(self, info_gain_variant='entropy', max_depth=None, random_forest=False, bagging=False, boosting=False, T=None, weights=None):
        self.root = None
        self.boosting = boosting
        self.bagging = bagging
        self.random_forest = random_forest
        self.T = T
        self.weights = weights
        self.decision_stumps = {}
        self.decision_stumps_error = {}
        self.bagged_trees = {}
        self.random_forest_trees = {}
        self.info_gain_variant = info_gain_variant
        self.max_depth = max_depth
        self.current_depth = 0

    def get_stumps_train_error(self):
        return self.decision_stumps_error
    
    def get_decision_stumps(self):
        return self.decision_stumps

    def get_bagged_trees(self):
        return self.bagged_trees
    
    def get_random_forest_trees(self):
        return self.random_forest_trees
    
    def fit(self, data_set, attributes, labels, feature_size=1, sample_frac=0.75, sample_size=0, replacement=True):
        dt_attributes = copy.deepcopy(attributes)
        dt_data_set = copy.deepcopy(data_set)
        for attribute in dt_attributes:

            if 'numeric' in dt_attributes[attribute]:
                self.handle_numeric_attribute(dt_data_set, attribute, dt_attributes)

        if self.boosting:
            self.max_depth = 1
            for i in range(0, self.T):
                # print('i: ', i)

                # # print('self.weights: ', self.weights)
                dt_data_set['weights'] = self.weights
                # print('data set weights: ', '\n', dt_data_set['weights'].head())
                # print('weights')
                # print(dt_data_set.head(20))
                # df = copy.deepcopy(dt_data_set)
                # df['play'] = labels
                # file_path = os.path.join(os.getcwd(), 'data1.csv')
                # df.to_csv(file_path)
                # print('starting w', weights_array)   

                # print('dt_data_set')
                # print(dt_data_set.head(20))
                # print('labels')
                # print(labels.head(20))
                #print 1st 5 weights
                self.root = self.build_tree_id3(dt_data_set, dt_attributes, copy.deepcopy(labels), curr_depth=0)
                error, predicted_labels = self.boosting_tree_error(dt_data_set, dt_attributes, copy.deepcopy(labels)) # todo
                # print('error: ', error)

                
                # print('predicted_labels')
                # print(predicted_labels.head(20))

                vote = self.boosting_tree_vote(error) # todo
                # print('vote: ', vote)

                # print('predicted_labels: ')
                # print(predicted_labels.head(20))
                new_weights = self.calculate_new_weights(self.weights, vote, copy.deepcopy(labels), predicted_labels) # todo
                self.weights = new_weights
                # print('new_weights: ')
                # print(self.weights.head(20))
      
                # print('self.weights: ', '\n', self.weights.head())
                self.decision_stumps[i] = [copy.deepcopy(self.root), vote, error]
                self.decision_stumps_error[i] = error
        elif self.bagging or self.random_forest:
            self.max_depth = float('inf')
            for i in range(0, self.T):

                if sample_size == 0:
                    sample_size = int(dt_data_set.shape[0] * sample_frac)
                data_set_sample = dt_data_set.sample(n=sample_size, replace=replacement)
                labels_sample = labels.loc[data_set_sample.index]

                self.root = self.build_tree_id3(data_set_sample, dt_attributes, labels_sample, curr_depth=0, feature_size=feature_size)
                if self.bagging:
                    self.bagged_trees[i] = copy.deepcopy(self.root)
                elif self.random_forest:
                    self.random_forest_trees[i] = copy.deepcopy(self.root)
        else:
            self.root = self.build_tree_id3(dt_data_set, dt_attributes, labels, curr_depth=0)

    def handle_numeric_attribute(self, data_set, attribute, attributes):
        madian_value = data_set[attribute].median()
        binary_values = [0, 1]
        data_set[attribute] = data_set[attribute].apply(lambda x: binary_values[0] if x < madian_value else binary_values[1])
        attributes[attribute] = binary_values

    def calculate_new_weights(self, weights, vote, labels, predicted_labels): # predicted_labels in -1, 1 format
        labels = labels.to_frame()

        new_weights = pd.DataFrame({'weights': []})
        for i in range(0, len(labels)):

            new_weights.loc[i] =  weights['weights'][i] * (np.exp((-1 * vote) * (labels.iloc[i][0] * predicted_labels.iloc[i][0])))
        
        normalization_factor = new_weights['weights'].sum()
        new_weights = new_weights.div(normalization_factor)
        # weights_array = np.array(new_weights['weights'])

        return new_weights
        
    def boosting_tree_error(self, dt_data_set, dt_attributes, labels):
        labels = labels.to_frame()
        predicted_labels = self.predict(dt_data_set, dt_attributes)

        error = 0
        for i in range(0, len(predicted_labels)):
            if predicted_labels.iloc[i].values != labels.iloc[i].values:
                error += dt_data_set.iloc[i]['weights']

        return error, predicted_labels

    def boosting_tree_vote(self, error):
        return 0.5 * (np.log((1 - error) / error)) # this equation breaks when error = 0 or error = 1, a small error term can be added to fix this
    
    def predict(self, data_set, attributes, num_trees=None, decision_tree=None):
            dt_attributes = copy.deepcopy(attributes)
            dt_data_set = copy.deepcopy(data_set)
            for attribute in dt_attributes:
                if 'numeric' in dt_attributes[attribute]:
                    self.handle_numeric_attribute(dt_data_set, attribute, dt_attributes)
            
            predicted_labels = pd.DataFrame({'label': []})
            
            for index, row in dt_data_set.iterrows():
                if num_trees != None:
                    if self.bagging:
                        predicted_labels.loc[index] = self.bagging_predict_row(row, self.bagged_trees, num_trees)
                    elif self.boosting:
                        predicted_labels.loc[index] = self.boosting_predict_row(row, self.decision_stumps, num_trees)
                    elif self.random_forest:
                        predicted_labels.loc[index] = self.bagging_predict_row(row, self.random_forest_trees, num_trees)
                else:
                    if decision_tree != None:
                        predicted_labels.loc[index] = self.predict_row(row, decision_tree)
                    elif self.root != None:
                        predicted_labels.loc[index] = self.predict_row(row, self.root)
 
            return predicted_labels

    def boosting_predict_row(self, row, decision_stumps, num_trees):
        prediction_total = 0

        sub_decision_stumps = dict(list(decision_stumps.items())[0:num_trees])
        for decision_stump in sub_decision_stumps:
            stump = sub_decision_stumps[decision_stump]
            stump_root = stump[0]
            stump_vote = stump[1]
            stump_prediction = self.predict_row(row, stump_root)
            prediction_total += (stump_vote * stump_prediction)
        return np.sign(prediction_total)

    def bagging_predict_row(self, row, bagged_trees, num_trees):
        predictions = 0
        sub_bagged_trees = dict(list(bagged_trees.items())[0:num_trees])
        for tree in sub_bagged_trees:
            tree_root = bagged_trees[tree]
            prediction = self.predict_row(row, tree_root)
            predictions += prediction
        return np.sign(predictions)

    def predict_row(self, row, node):
        if node.label != None:
            return node.label
        else:
            attribute_value = row[node.attribute]
            next_node = node.children[attribute_value]
            return self.predict_row(row, next_node)


    def get_best_attribute(self, data_set, attributes, labels, boosting):
        if self.info_gain_variant == 'me':
            # print('get_best_attribute: me')
            return information_gain_me(data_set, attributes, labels)
        elif self.info_gain_variant == 'gini':
            # print('get_best_attribute: gini')
            return information_gain_gini(data_set, attributes, labels)
        else:
            # print('get_best_attribute: entropy')
            return information_gain_entropy(data_set, attributes, labels, boosting)
    

    # ID3 algorithm
    def build_tree_id3(self, data_set, attributes, labels, curr_depth, feature_size=1):

        if len(set(labels)) == 1: 
            similar_label = labels.iloc[0]
            return DecisionTreeNode(label=similar_label)
        elif len(attributes) == 0:  
            most_common_label = labels.value_counts().idxmax()
            return DecisionTreeNode(label=most_common_label)
        else:
            root = DecisionTreeNode()
            data_set_attributes = {}

            if self.random_forest:
                random_attributes = random.sample(list(attributes.keys()), feature_size)
                data_set_attributes = {k: attributes[k] for k in random_attributes}
            else:
                data_set_attributes = attributes

            best_attribute_dict = self.get_best_attribute(data_set, data_set_attributes, labels, boosting=self.boosting)
            best_attribute =  next(iter(best_attribute_dict))
            root.attribute = best_attribute
            root.info_gain = best_attribute_dict[best_attribute]

            if curr_depth + 1 == self.max_depth:
                for attribute_value in attributes[best_attribute]:
                    attribute_value_subset = data_set[data_set[best_attribute] == attribute_value]
                    matching_rows = labels.index.isin(attribute_value_subset.index) 
                    labels_subset = labels[matching_rows]



                    if len(attribute_value_subset) == 0:
                        most_common_label = None
                        if self.boosting:
                            most_common_label = self.get_weighted_most_freq_label(data_set, labels_subset)
                        else:
                            most_common_label = labels.value_counts().idxmax()
                        root.add_child(attribute_value, DecisionTreeNode(label=most_common_label))
                    else:
                        most_freq_label = None


                        if self.boosting:
                            most_freq_label = self.get_weighted_most_freq_label(attribute_value_subset, labels_subset)
                        else:
                            most_freq_label = labels_subset.value_counts().idxmax()

                        root.add_child(attribute_value, DecisionTreeNode(label=most_freq_label))
            else:
                for attribute_value in attributes[best_attribute]:
                    attribute_value_subset = data_set[data_set[best_attribute] == attribute_value] ## 
                    matching_rows = labels.index.isin(attribute_value_subset.index) 
                    labels_subset = labels[matching_rows]

                    if len(attribute_value_subset) == 0: 
                        most_common_label = labels.value_counts().idxmax()
                        root.add_child(attribute_value, DecisionTreeNode(label=most_common_label))
                    else:
                        updated_attributes = attributes.copy()
                        del updated_attributes[best_attribute]
                        sub_tree = self.build_tree_id3(attribute_value_subset, updated_attributes, labels_subset, curr_depth+1)
                        root.add_child(attribute_value, sub_tree)
            return root
        
    def get_weighted_most_freq_label(self, data_set, labels):
        label_weights = {}
        for label in set(labels):
            label_weights[label] = 0
            label_indices = np.where(labels == label)[0]
            for index in label_indices:
                label_weights[label] += data_set.iloc[index, -1]

        return max(label_weights, key=label_weights.get)