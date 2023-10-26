import math

def information_gain_entropy(data_set, attributes, labels, boosting):
    # Return the attribute with the highest information gain
    # according to the entropy measure
    weights_name = ''
    data_set_proportions = []
    total_instances = 0
    labels = labels.to_frame()
    label_column_name = labels.columns[0]

    # print('boosting', boosting)
    if boosting:
        # print('boosting:14 ', boosting)
        # label_name = data_set.columns[-2] # second to last column because last column is weights
        weights_name = data_set.columns[-1]
        data_set_proportions = get_boosting_proportions(data_set, labels, label_column_name, weights_name) # get proportions from weights
        # print('data_set_proportions18', data_set_proportions)
        total_instances = 1
        # print('data_set_proportions', data_set_proportions)
    else:
        # label_name = data_set.columns[-1]
        # data_set_proportions = labels.groupby(label_column_name).size().div(len(labels)).values
        data_set_proportions = get_proportions(data_set, labels, label_column_name)
        total_instances = len(data_set)

    total_data_set_entropy = calculate_gini(data_set_proportions)
    features_entropy = {}

    # print('data_set_proportions29', data_set_proportions)
    # print('total_data_set_entropy30', total_data_set_entropy)
    # print('total_instances31', total_instances)
    # print('\n')
    
    gains = {}

    for attribute in attributes:
        attribute_expected_entropy = 0
        attribute_entropy = 0
        # print('attribute38', attribute)

        for attribute_value in attributes[attribute]:
            # print('attribute_value', attribute_value)
            attribute_value_subset = data_set[data_set[attribute] == attribute_value]
            total_subset_instances = 0

            if boosting:
                total_subset_instances = attribute_value_subset[weights_name].sum()
            else:
                total_subset_instances = len(attribute_value_subset)

            attribute_value_subset_proportions = []

            if boosting:

                attribute_value_subset_proportions = get_boosting_proportions(attribute_value_subset, labels, label_column_name, weights_name) # get proportions from weights

            else:
                attribute_value_subset_proportions = get_proportions(attribute_value_subset, labels, label_column_name)
            # print('attribute_value_subset_proportions', attribute_value_subset_proportions)

            attribute_value_subset_entropy = calculate_gini(attribute_value_subset_proportions)
            # print('attribute_value_subset_entropy', attribute_value_subset_entropy)
            attribute_entropy += attribute_value_subset_entropy
            attribute_expected_entropy += (attribute_value_subset_entropy * (total_subset_instances / total_instances))
        #     print('total_subset_instances', total_subset_instances)
        #     print('total_instances', total_instances)
        #     print('contribution', (attribute_value_subset_entropy * (total_subset_instances / total_instances)))
        # print('attribute_expected_entropy', attribute_expected_entropy)
        gains[attribute] = total_data_set_entropy - attribute_expected_entropy
        features_entropy[attribute] = attribute_entropy
    
    best_attribute_dict = {}
    best_attribute = max(gains, key=gains.get)

    best_attribute_gain = gains[best_attribute]
    # print('best_attribute', best_attribute)
    # print('best_attribute_gain', best_attribute_gain, '\n' )
    best_attribute_dict[best_attribute] = best_attribute_gain

    # print('features_entropy', features_entropy)

    lowest_entropy_feature = min(features_entropy, key=features_entropy.get)
    lowest_entropy = features_entropy[lowest_entropy_feature]
    # print('lowest_entropy_feature', lowest_entropy_feature, 'lowest_entropy', lowest_entropy)
    # # best_attribute_dict[lowest_entropy_feature] = lowest_entropy
    # print('best_attribute_dict', best_attribute_dict)
    # print('features_entropy', features_entropy, '\n')

    return best_attribute_dict
    

def calculate_entropy(proportions):
    for proportion in proportions:
      if proportion == 0:
        return 0
    entropy = 0
    for proportion in proportions:
      entropy += (-((proportion) * math.log2(proportion)))
    return entropy

def information_gain_gini(data_set, attributes, labels):
    # Return the attribute with the highest information gain
    # according to the gini measure
    label_name = data_set.columns[-1]
    data_set_proportions = data_set.groupby(label_name).size().div(len(data_set)).values
    total_data_set_entropy = calculate_gini(data_set_proportions)
    # print('total_data_set_entropy', total_data_set_entropy)
    total_instances = len(data_set)
    gains = {}

    for attribute in attributes:
        attribute_expected_entropy = 0
        for attribute_value in attributes[attribute]:
            attribute_value_subset = data_set[data_set[attribute] == attribute_value]
            total_subset_instances = len(attribute_value_subset)
            attribute_value_subset_proportions = attribute_value_subset.groupby(label_name).size().div(len(attribute_value_subset)).values
            attribute_value_subset_entropy = calculate_gini(attribute_value_subset_proportions)
            attribute_expected_entropy += (attribute_value_subset_entropy * (total_subset_instances / total_instances))
        gains[attribute] = total_data_set_entropy - attribute_expected_entropy
    best_attribute_dict = {}
    best_attribute = max(gains, key=gains.get)   
    best_attribute_gain = gains[best_attribute]
    # print('best_attribute', best_attribute)
    # print('best_attribute_gain', best_attribute_gain, '\n' )
    best_attribute_dict[best_attribute] = best_attribute_gain
    
    return best_attribute_dict
    
def calculate_gini(proportions):
    gini_index = 0
    x = 0
    for proportion in proportions:
      x += proportion**2
    gini_index = 1 - x
    return gini_index

def information_gain_me(data_set, attributes, labels):
    # Return the attribute with the highest information gain
    # according to the majority error measure
    label_name = data_set.columns[-1]
    data_set_proportions = data_set.groupby(label_name).size().div(len(data_set)).values
    #get max proportion
    max_proportion = max(data_set_proportions)

    total_data_set_entropy = 1 - max_proportion
    # print('total_data_set_entropy', total_data_set_entropy)
    total_instances = len(data_set)
    gains = {}

    for attribute in attributes:
        attribute_expected_error = 0
        for attribute_value in attributes[attribute]:
            attribute_value_subset = data_set[data_set[attribute] == attribute_value]
            if len(attribute_value_subset) == 0:
                continue

            total_subset_instances = len(attribute_value_subset)
            most_frequent_value = attribute_value_subset[label_name].value_counts().max() / len(attribute_value_subset)
            error_rate = 1 - most_frequent_value

            # attribute_value_subset_proportions = attribute_value_subset.groupby(label_name).size().div(len(attribute_value_subset)).values
            # attribute_value_subset_entropy = calculate_entropy(attribute_value_subset_proportions)
            attribute_expected_error += (error_rate * (total_subset_instances / total_instances))
        gains[attribute] = total_data_set_entropy - attribute_expected_error
    
    best_attribute_dict = {}
    best_attribute = max(gains, key=gains.get)   
    best_attribute_gain = gains[best_attribute]
    # print('best_attribute', best_attribute)
    # print('best_attribute_gain', best_attribute_gain, '\n' )
    best_attribute_dict[best_attribute] = best_attribute_gain
    
    return best_attribute_dict

def get_proportions(data_set, labels, labels_column_name):
    proportions = []
    matching_rows = labels.index.isin(data_set.index) 
    labels_subset = labels[matching_rows]

    grouped_data = labels_subset.groupby(labels_column_name)

    for group in grouped_data.groups:
        sub_group_len = len(grouped_data.get_group(group))
        proportions.append(sub_group_len / len(labels_subset))


    return proportions

def get_boosting_proportions(data_set, labels, labels_column_name, weights_name):
    proportions = []
    subset_total = 0
    matching_rows = labels.index.isin(data_set.index) 
    labels_subset = labels[matching_rows]

    grouped_data = labels_subset.groupby(labels_column_name)
    for group in grouped_data.groups:
        # print('group', group)
        group_indices = grouped_data.get_group(group).index
        # print('group_indices', group_indices)
        sub_data = data_set.loc[group_indices]
        sum_of_weights = sub_data[weights_name].sum()
        subset_total += sum_of_weights
        proportions.append(sum_of_weights)
    proportions = [x / subset_total for x in proportions]

    return proportions