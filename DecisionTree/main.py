import pandas as pd
import re
import copy
from decision_tree import DecisionTree


# 2. [30 points] We will implement a decision tree learning algorithm for car evaluation task.
print('2. [30 points] We will implement a decision tree learning algorithm for car evaluation task.')
car_attributes = {}
column_names = []

# Read the file and split it into sections
with open('data/car-4/data-desc.txt', 'r') as file:
    sections = file.read().split('|')

    attributes_section = sections[2].strip().split('\n')
    attributes_section = attributes_section[1:] # Skip the first element
    for line in attributes_section:
        attribute, values = line.strip('.').split(':')
        attribute = attribute.strip()
        values = [value.strip() for value in values.split(',')]
        car_attributes[attribute] = values

    column_names = sections[3].strip()
    column_names = re.split(r'\n|,', column_names)
    column_names = column_names[1:] # Skip the first element


car_training_data = pd.read_csv('data/car-4/train.csv', header=None, names=column_names)
car_testing_data = pd.read_csv('data/car-4/test.csv', header=None, names=column_names)

car_training_labels = car_training_data.iloc[:, -1]
car_testing_labels = car_testing_data.iloc[:, -1]

# 2. (b) [10 points] Use your implemented algorithm to learn decision trees from the training data.
print('2. (b) [10 points] Use your implemented algorithm to learn decision trees from the training data.')

# Build different decision trees with different max_depth
decision_trees = {}
print('Building Decision Trees with depth 1 to 6')
for max_depth in range(1, 7):
    print('Build Tree with Max Depth: ', max_depth)
    decision_tree = DecisionTree(max_depth=max_depth)
    decision_tree.fit(car_training_data, car_attributes, car_training_labels)
    decision_trees[max_depth] = copy.deepcopy(decision_tree)

# Test decision trees with different max_depth on testing and training data
variants = ['entropy', 'me', 'gini']
decision_trees_errors_train = {}
decision_trees_errors_test = {}
for max_depth in range(1, 7):
    print('Testing Tree with Max Depth: ', max_depth)
    decision_tree = decision_trees[max_depth]
    total_error_train = 0
    total_error_test = 0
    
    for variant in variants:
        print('                         Testing Tree with Variant: ', variant)
        decision_tree.info_gain_variant = variant
        predictions_train = decision_tree.predict(car_training_data, car_attributes)
        predictions_test = decision_tree.predict(car_testing_data, car_attributes)

        error_train = (predictions_train.label != car_training_labels).sum() / len(car_training_labels)
        error_test = (predictions_test.label != car_testing_labels).sum() / len(car_testing_labels)

        total_error_train += error_train
        total_error_test += error_test
    decision_trees_errors_train[max_depth] = total_error_train / len(variants)
    decision_trees_errors_test[max_depth] = total_error_test / len(variants)
    


# Create a dataframe with the errors
df_train = pd.DataFrame({'Max Depth': list(decision_trees_errors_train.keys()), ' Error': list(decision_trees_errors_train.values())})
df_train.set_index('Max Depth', inplace=True)

df_test = pd.DataFrame({'Max Depth': list(decision_trees_errors_test.keys()), ' Error': list(decision_trees_errors_test.values())})
df_test.set_index('Max Depth', inplace=True)

print('Average Error vs Max Depth: Training')
print(df_train, '\n')
print('Average Error vs Max Depth: Testing')
print(df_test, '\n')

# plot_train = df_train.plot(kind='bar', title='Error vs Max Depth: Training', grid=False, legend=False, xlabel='Max Depth', ylabel='Error')
# # plot_train.grid(axis='x')

# plot_test = df_test.plot(kind='bar', title='Error vs Max Depth: Testing', grid=False, legend=False, xlabel='Max Depth', ylabel='Error')

#3. [25 points] Next, modify your implementation a little bit to support numerical attributes. 
print('3. [25 points] Next, modify your implementation a little bit to support numerical attributes.')


bank_column_names = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
bank_attributes = {
    'age': ['numeric'],
    'job': ['admin.', 'unknown', 'unemployed', 'management', 'housemaid', 'entrepreneur', 'student', 'blue-collar', 'self-employed', 'retired', 'technician', 'services'],
    'marital': ['married', 'divorced', 'single'],
    'education': ['unknown', 'secondary', 'primary', 'tertiary'],
    'default': ['yes', 'no'],
    'balance': ['numeric'],
    'housing': ['yes', 'no'],
    'loan': ['yes', 'no'],
    'contact': ['unknown', 'telephone', 'cellular'],
    'day': ['numeric'],
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'duration': ['numeric'],
    'campaign': ['numeric'],
    'pdays': ['numeric'],
    'previous': ['numeric'],
    'poutcome': ['unknown', 'other', 'failure', 'success'],
}


bank_training_data = pd.read_csv('data/bank-4/train.csv', names=bank_column_names)
bank_testing_data = pd.read_csv('data/bank-4/test.csv', names=bank_column_names)
bank_training_labels = bank_training_data.iloc[:, -1]
bank_testing_labels = bank_testing_data.iloc[:, -1]

# 3. (a) [10 points] Let us consider “unknown” as a particular attribute value, and hence we do not have any missing attributes for both training and test. 

bank_decision_trees = {}
print('Building Decision Trees with depth 1 to 16')
for max_depth in range(1, 17):
    print('Build Tree with Max Depth: ', max_depth)
    decision_tree = DecisionTree(max_depth=max_depth)
    decision_tree.fit(bank_training_data, bank_attributes, bank_training_labels)
    bank_decision_trees[max_depth] = copy.deepcopy(decision_tree)

# Test decision trees with different max_depth on testing and training data
bank_variants = ['entropy', 'me', 'gini']
bank_decision_trees_errors_train = {}
bank_decision_trees_errors_test = {}
for max_depth in range(1, 17):
    print('Testing Tree with Max Depth: ', max_depth)
    decision_tree = bank_decision_trees[max_depth]
    bank_total_error_train = 0
    bank_total_error_test = 0

    for variant in bank_variants:
        print('                         Testing Tree with Variant: ', variant)
        decision_tree.info_gain_variant = variant
        bank_predictions_train = decision_tree.predict(bank_training_data, bank_attributes)
        bank_predictions_test = decision_tree.predict(bank_testing_data, bank_attributes)

        bank_error_train = (bank_predictions_train.label != bank_training_labels).sum() / len(bank_training_labels)
        bank_error_test = (bank_predictions_test.label != bank_testing_labels).sum() / len(bank_testing_labels)

        bank_total_error_train += bank_error_train
        bank_total_error_test += bank_error_test
    bank_decision_trees_errors_train[max_depth] = bank_total_error_train / len(bank_variants)
    bank_decision_trees_errors_test[max_depth] = bank_total_error_test / len(bank_variants)
    


# Create a dataframe with the errors
df_bank_train = pd.DataFrame({'Max Depth': list(bank_decision_trees_errors_train.keys()), ' Error': list(bank_decision_trees_errors_train.values())})
df_bank_train.set_index('Max Depth', inplace=True)

df_bank_test = pd.DataFrame({'Max Depth': list(bank_decision_trees_errors_test.keys()), ' Error': list(bank_decision_trees_errors_test.values())})
df_bank_test.set_index('Max Depth', inplace=True)

print('Average Error vs Max Depth: Training')
print(df_bank_train, '\n')
print('Average Error vs Max Depth: Testing')
print(df_bank_test, '\n')

# plot_bank_train = df_bank_train.plot(kind='bar', title='Error vs Max Depth: Training', grid=False, legend=False, xlabel='Max Depth', ylabel='Error')
# # plot_bank_train.grid(axis='x')

# plot_bank_test = df_bank_test.plot(kind='bar', title='Error vs Max Depth: Testing', grid=False, legend=False, xlabel='Max Depth', ylabel='Error')

#3. (b) [10 points] Let us consider ”unknown” as attribute value missing.
print('3. (b) [10 points] Let us consider ”unknown” as attribute value missing.')
bank_attributes_2 = copy.deepcopy(bank_attributes)

bank_training_data_2 = copy.deepcopy(bank_training_data)
bank_testing_data_2 = copy.deepcopy(bank_testing_data)

bank_training_labels_2 = bank_training_data_2.iloc[:, -1]
bank_testing_labels_2 = bank_testing_data_2.iloc[:, -1]

#Replace unknown values with the most common value in the column
for col in bank_training_data_2.columns:
    if bank_training_data_2[col].isin(['unknown']).any():
        value_counts = bank_training_data_2[col].value_counts()         
        most_common_value = value_counts.drop('unknown').idxmax()
        bank_training_data_2[col] = bank_training_data_2[col].replace('unknown', most_common_value)

    if bank_testing_data_2[col].isin(['unknown']).any():
        value_counts = bank_testing_data_2[col].value_counts()         
        most_common_value = value_counts.drop('unknown').idxmax()
        bank_testing_data_2[col] = bank_testing_data_2[col].replace('unknown', most_common_value)

for attribute in bank_attributes_2:
    if 'unknown' in bank_attributes_2[attribute]:
        bank_attributes_2[attribute].remove('unknown')

bank_decision_trees_2 = {}
print('Building Decision Trees with depth 1 to 16')
for max_depth in range(1, 17):
    print('Build Tree with Max Depth: ', max_depth)
    decision_tree = DecisionTree(max_depth=max_depth)
    decision_tree.fit(bank_training_data_2, bank_attributes_2, bank_training_labels_2)
    bank_decision_trees_2[max_depth] = copy.deepcopy(decision_tree)


# Test decision trees with different max_depth on testing and training data
bank_variants_2 = ['entropy', 'me', 'gini']
bank_decision_trees_errors_train_2 = {}
bank_decision_trees_errors_test_2 = {}
for max_depth in range(1, 17):
    print('Testing Trees with Max Depth: ', max_depth)
    decision_tree = bank_decision_trees_2[max_depth]
    bank_total_error_train = 0
    bank_total_error_test = 0

    for variant in bank_variants_2:
        print('                         Testing Tree with Variant: ', variant)
        decision_tree.info_gain_variant = variant
        bank_predictions_train_2 = decision_tree.predict(bank_training_data_2, bank_attributes_2)
        bank_predictions_test_2 = decision_tree.predict(bank_testing_data_2, bank_attributes_2)

        bank_error_train = (bank_predictions_train_2.label != bank_training_labels_2).sum() / len(bank_training_labels_2)
        bank_error_test = (bank_predictions_test_2.label != bank_testing_labels_2).sum() / len(bank_testing_labels_2)

        bank_total_error_train += bank_error_train
        bank_total_error_test += bank_error_test
    bank_decision_trees_errors_train_2[max_depth] = bank_total_error_train / len(bank_variants_2)
    bank_decision_trees_errors_test_2[max_depth] = bank_total_error_test / len(bank_variants_2)



# Create a dataframe with the errors
df_bank_train_2 = pd.DataFrame({'Max Depth': list(bank_decision_trees_errors_train_2.keys()), ' Error': list(bank_decision_trees_errors_train_2.values())})
df_bank_train_2.set_index('Max Depth', inplace=True)

df_bank_test_2 = pd.DataFrame({'Max Depth': list(bank_decision_trees_errors_test_2.keys()), ' Error': list(bank_decision_trees_errors_test_2.values())})
df_bank_test_2.set_index('Max Depth', inplace=True)

print('Average Error vs Max Depth: Training')
print(df_bank_train_2, '\n')
print('Average Error vs Max Depth: Testing')
print(df_bank_test_2, '\n')


# plot_bank_train_2 = df_bank_train_2.plot(kind='bar', title='Error vs Max Depth: Training', grid=False, legend=False, xlabel='Max Depth', ylabel='Error')
# # plot_bank_train.grid(axis='x')

# plot_bank_test_2 = df_bank_test_2.plot(kind='bar', title='Error vs Max Depth: Testing', grid=False, legend=False, xlabel='Max Depth', ylabel='Error')














