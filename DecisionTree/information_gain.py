import math

def information_gain_entropy(data_set, attributes, labels):
    # Return the attribute with the highest information gain
    # according to the entropy measure
    label_name = data_set.columns[-1]
    data_set_proportions = data_set.groupby(label_name).size().div(len(data_set)).values
    total_data_set_entropy = calculate_entropy(data_set_proportions)
    total_instances = len(data_set)
    gains = {}

    for attribute in attributes:
        attribute_expected_entropy = 0
        for attribute_value in attributes[attribute]:
            attribute_value_subset = data_set[data_set[attribute] == attribute_value]
            total_subset_instances = len(attribute_value_subset)
            attribute_value_subset_proportions = attribute_value_subset.groupby(label_name).size().div(len(attribute_value_subset)).values
            attribute_value_subset_entropy = calculate_entropy(attribute_value_subset_proportions)
            attribute_expected_entropy += (attribute_value_subset_entropy * (total_subset_instances / total_instances))
        gains[attribute] = total_data_set_entropy - attribute_expected_entropy
    return max(gains, key=gains.get)
    

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
    total_data_set_entropy = calculate_entropy(data_set_proportions)
    total_instances = len(data_set)
    gains = {}

    for attribute in attributes:
        attribute_expected_entropy = 0
        for attribute_value in attributes[attribute]:
            attribute_value_subset = data_set[data_set[attribute] == attribute_value]
            total_subset_instances = len(attribute_value_subset)
            attribute_value_subset_proportions = attribute_value_subset.groupby(label_name).size().div(len(attribute_value_subset)).values
            attribute_value_subset_entropy = calculate_entropy(attribute_value_subset_proportions)
            attribute_expected_entropy += (attribute_value_subset_entropy * (total_subset_instances / total_instances))
        gains[attribute] = total_data_set_entropy - attribute_expected_entropy
    return max(gains, key=gains.get)   
    


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
    total_data_set_entropy = calculate_entropy(data_set_proportions)
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
    return max(gains, key=gains.get)   