import pandas as pd
import re
import copy
from decision_tree import DecisionTree



#2. (b) [10 points]
#Use your implemented algorithm to learn decision trees from the training data. 
# Vary the maximum tree depth from 1 to 6 — for each setting, run your algorithm to learn a decision tree, 
# and use the tree to predict both the training and test examples.

car_attributes = {}
column_names = []

# Read the file and split it into sections
with open('../data/car-4/data-desc.txt', 'r') as file:
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


car_training_data = pd.read_csv('../data/car-4/train.csv', header=None, names=column_names)
car_testing_data = pd.read_csv('../data/car-4/test.csv', header=None, names=column_names)

car_training_labels = car_training_data.iloc[:, -1]
car_testing_labels = car_testing_data.iloc[:, -1]

# Build different decision trees with different max_depth
car_decision_trees = {}
dt_car_variants = {}
variants = ['entropy', 'me', 'gini']
for variant in variants:
    print('Building Decision Trees with depth 1 to 6 (variant: ', variant, ')')
    for max_depth in range(1, 7):
        print('Build Tree with Max Depth: ', max_depth)
        decision_tree = DecisionTree(info_gain_variant=variant, max_depth=max_depth)
        decision_tree.fit(car_training_data, car_attributes, car_training_labels)
        car_decision_trees[max_depth] = copy.deepcopy(decision_tree)
    dt_car_variants[variant] = copy.deepcopy(car_decision_trees)

##
car_restults_train = {}
car_restults_test = {}

for variant in dt_car_variants:
    total_error_train = 0
    total_error_test = 0
    print('Testing Tree with Variant: ', variant)
    for max_depth in dt_car_variants[variant]:
        print('                         Testing Tree with Max Depth: ', max_depth)
        dt = dt_car_variants[variant][max_depth]
        prediction_train = dt.predict(car_training_data, car_attributes)
        prediction_test = dt.predict(car_testing_data, car_attributes)
        error_train = (prediction_train.label != car_training_labels).sum() / len(car_training_labels)
        error_test = (prediction_test.label != car_testing_labels).sum() / len(car_testing_labels)
        total_error_train += error_train
        total_error_test += error_test
    car_restults_train[variant] = total_error_train / len(dt_car_variants[variant])
    car_restults_test[variant] = total_error_test / len(dt_car_variants[variant])


dt_car_restults_train = pd.DataFrame({'Heuristic': list(car_restults_train.keys()), 'Average Error': list(car_restults_train.values())})
dt_car_restults_train.columns = ['Heuristic', 'Average Error (Train)']

dt_car_restults_test = pd.DataFrame({'Heuristic': list(car_restults_test.keys()), 'Average Error': list(car_restults_test.values())})
dt_car_restults_test.columns = ['Heuristic', 'Average Error (Test)']

dt_car_restults_train['Heuristic'] = dt_car_restults_train['Heuristic'].replace(['entropy', 'me', 'gini'], ['Information Gain', 'Majority Error', 'Gini Index'])

dt_car_restults_test['Heuristic'] = dt_car_restults_test['Heuristic'].replace(['entropy', 'me', 'gini'], ['Information Gain', 'Majority Error', 'Gini Index'])




# 3. (a) [10 points] Let us consider “unknown” as a particular attribute value, 
# and hence we do not have any missing attributes for both training and test. 
# Vary the maximum tree depth from 1 to 16 — for each setting, run your algorithm to 
# learn a decision tree, and use the tree to predict both the training and test examples.

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


bank_training_data = pd.read_csv('../data/bank-4/train.csv', names=bank_column_names)
bank_testing_data = pd.read_csv('../data/bank-4/test.csv', names=bank_column_names)
bank_training_labels = bank_training_data.iloc[:, -1]
bank_testing_labels = bank_testing_data.iloc[:, -1]


# Build different decision trees with different max_depth
bank_decision_trees = {}
dt_bank_variants = {}
variants = ['entropy', 'me', 'gini']
for variant in variants:
    print('Building Decision Trees with depth 1 to 16 (variant: ', variant, ')')
    for max_depth in range(1, 17):
        print('Build Tree with Max Depth: ', max_depth)
        decision_tree = DecisionTree(info_gain_variant=variant, max_depth=max_depth)
        decision_tree.fit(bank_training_data, bank_attributes, bank_training_labels)
        bank_decision_trees[max_depth] = copy.deepcopy(decision_tree)
    dt_bank_variants[variant] = copy.deepcopy(bank_decision_trees)



bank_restults_train = {}
bank_restults_test = {}

for variant in dt_bank_variants:
    total_error_train = 0
    total_error_test = 0
    print('Testing Tree with Variant: ', variant)
    for max_depth in dt_bank_variants[variant]:
        print('                         Testing Tree with Max Depth: ', max_depth)
        dt = dt_bank_variants[variant][max_depth]
        bank_prediction_train = dt.predict(bank_training_data, bank_attributes)
        bank_prediction_test = dt.predict(bank_testing_data, bank_attributes)
        error_train = (bank_prediction_train.label != bank_training_labels).sum() / len(bank_training_labels)
        error_test = (bank_prediction_test.label != bank_testing_labels).sum() / len(bank_testing_labels)
        total_error_train += error_train
        total_error_test += error_test
    bank_restults_train[variant] = total_error_train / len(dt_bank_variants[variant])
    bank_restults_test[variant] = total_error_test / len(dt_bank_variants[variant])

dt_bank_restults_train = pd.DataFrame({'Heuristic': list(bank_restults_train.keys()), 'Average Error': list(bank_restults_train.values())})
dt_bank_restults_train.columns = ['Heuristic', 'Average Error (Train)']

dt_bank_restults_test = pd.DataFrame({'Heuristic': list(bank_restults_test.keys()), 'Average Error': list(bank_restults_test.values())})
dt_bank_restults_test.columns = ['Heuristic', 'Average Error (Test)']

dt_bank_restults_train['Heuristic'] = dt_bank_restults_train['Heuristic'].replace(['entropy', 'me', 'gini'], ['Information Gain', 'Majority Error', 'Gini Index'])
dt_bank_restults_test['Heuristic'] = dt_bank_restults_test['Heuristic'].replace(['entropy', 'me', 'gini'], ['Information Gain', 'Majority Error', 'Gini Index'])




# 3. (b) [10 points] Let us consider ”unknown” as attribute value missing.
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

# Build different decision trees with different max_depth
bank_decision_trees_2 = {}
dt_bank_variants_2 = {}
variants = ['entropy', 'me', 'gini']
for variant in variants:
    print('Building Decision Trees with depth 1 to 16 (variant: ', variant, ')')
    for max_depth in range(1, 17):
        print('Build Tree with Max Depth: ', max_depth)
        decision_tree = DecisionTree(info_gain_variant=variant, max_depth=max_depth)
        decision_tree.fit(bank_training_data_2, bank_attributes_2, bank_training_labels_2)
        bank_decision_trees_2[max_depth] = copy.deepcopy(decision_tree)
    dt_bank_variants_2[variant] = copy.deepcopy(bank_decision_trees_2)


bank_restults_train_2 = {}
bank_restults_test_2 = {}

for variant in dt_bank_variants_2:
    total_error_train = 0
    total_error_test = 0
    print('Testing Tree with Variant: ', variant)
    for max_depth in dt_bank_variants_2[variant]:
        print('                         Testing Tree with Max Depth: ', max_depth)
        dt = dt_bank_variants_2[variant][max_depth]
        prediction_train = dt.predict(bank_training_data_2, bank_attributes_2)
        prediction_test = dt.predict(bank_testing_data_2, bank_attributes_2)
        error_train = (prediction_train.label != bank_training_labels_2).sum() / len(bank_training_labels_2)
        error_test = (prediction_test.label != bank_testing_labels_2).sum() / len(bank_testing_labels_2)
        total_error_train += error_train
        total_error_test += error_test
    bank_restults_train_2[variant] = total_error_train / len(dt_bank_variants_2[variant])
    bank_restults_test_2[variant] = total_error_test / len(dt_bank_variants_2[variant])

df_bank_variants_train_2 = pd.DataFrame({'Heuristic': list(bank_restults_train_2.keys()), 'Average Error': list(bank_restults_train_2.values())})
df_bank_variants_train_2.columns = ['Heuristic', 'Average Error (Train)']

df_bank_variants_test_2 = pd.DataFrame({'Heuristic': list(bank_restults_test_2.keys()), 'Average Error': list(bank_restults_test_2.values())})
df_bank_variants_test_2.columns = ['Heuristic', 'Average Error (Test)']

df_bank_variants_train_2['Heuristic'] = df_bank_variants_train_2['Heuristic'].replace(['entropy', 'me', 'gini'], ['Information Gain', 'Majority Error', 'Gini Index'])
df_bank_variants_test_2['Heuristic'] = df_bank_variants_test_2['Heuristic'].replace(['entropy', 'me', 'gini'], ['Information Gain', 'Majority Error', 'Gini Index'])


print('2. (b)')
print('\n\n')
print(dt_car_restults_train, '\n\n')
print(dt_car_restults_test, '\n\n')

print('3. (a)')
print('\n\n')
print(dt_bank_restults_train, '\n\n')
print(dt_bank_restults_test, '\n\n')

print('3. (b)')
print('\n\n')
print(df_bank_variants_train_2, '\n\n')
print(df_bank_variants_test_2, '\n\n')

