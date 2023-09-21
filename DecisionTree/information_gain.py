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
    pass

def information_gain_me(data_set, attributes, labels):
    # Return the attribute with the highest information gain
    # according to the majority error measure
    pass