This is a machine learning library develop by Osee Pierre for University of Utah's CS5350 course.

    Parameters:
    -----------
    info_gain_variant : str, optional (default='entropy')
        The variant of information gain to use. Can be either 'entropy', 'gini' or 'me'.
    max_depth : int, optional (default=None)
        The maximum depth of the decision tree. If None, the tree will be grown until all leaves are pure.
    random_forest : bool, optional (default=False)
        Whether to use random forest instead of decision tree.
    bagging : bool, optional (default=False)
        Whether to use bagging instead of decision tree.
    boosting : bool, optional (default=False)
        Whether to use boosting instead of decision tree.
    T : int, optional (default=None)
        The number of iterations to run for boosting.
    weights : pandas.DataFrame, optional (default=None)
        The weights for each data point in the training set.

    Methods:
    --------
    get_stumps_train_error()
        Returns the training error for each decision stump.
    get_decision_stumps()
        Returns the decision stumps learned during boosting.
    get_bagged_trees()
        Returns the trees learned during bagging.
    get_random_forest_trees()
        Returns the trees learned during random forest.
    fit(data_set, attributes, labels, feature_size=1, sample_frac=0.75, sample_size=0, replacement=True)
        Fits the decision tree to the given training set.
    predict(data_set, attributes, num_trees=None, decision_tree=None)
    """

Using the `DecisionTree` class:

```python
from decision_tree import DecisionTree

# Load the data from a file
train = pd.read_csv('..train.csv', names=bank_column_names)
test = pd.read_csv('..test.csv', names=bank_column_names)
attributes = {'at1': ['value1'],
    'at2': ['value1.', 'value2', 'value3', 'value4']}

X_train = train.drop('y', axis=1)
y_train = train['y']
y_train = y_train.replace(('yes', 'no'), (1, -1))

X_test = test.drop('y', axis=1)
y_test = test['y']
y_test = y_test.replace(('yes', 'no'), (1, -1))



# Create a decision tree learner
dt = DecisionTree(info_gain_variant='gini', max_depth=5)

# Learn a decision tree from the training data
dt.fit(X_train, attributes, y_train)

# Evaluate the decision tree on the testing data
prediction = dt.predict(test_data, attributes)

# error
error = (prediction.label != y_train).sum() / len(y_train)


```

Using the `DecisionTree` class with bagging:

```python
from decision_tree import DecisionTree

# Load the data from a file
train = pd.read_csv('..train.csv', names=bank_column_names)
test = pd.read_csv('..test.csv', names=bank_column_names)
attributes = {'at1': ['value1'],
    'at2': ['value1.', 'value2', 'value3', 'value4']}

X_train = train.drop('y', axis=1)
y_train = train['y']
y_train = y_train.replace(('yes', 'no'), (1, -1))

X_test = test.drop('y', axis=1)
y_test = test['y']
y_test = y_test.replace(('yes', 'no'), (1, -1))

# Create a decision tree learner
dt = DecisionTree(bagging=True,  T=10)

# Learn a decision tree from the training data
dt.fit(X_train, attributes, y_train, sample_frac=1, replacement=True)

# Evaluate the decision tree on the testing data
# indicate how many of the generated trees to use with parameter: num_trees
prediction = dt.predict(test_data, attributes, num_trees=10)

# error
error = (prediction.label != y_train).sum() / len(y_train)

```

Using the `DecisionTree` class with boosting:

```python

from decision_tree import DecisionTree

# Load the data from a file
train = pd.read_csv('..train.csv', names=bank_column_names)
test = pd.read_csv('..test.csv', names=bank_column_names)
attributes = {'at1': ['value1'],
    'at2': ['value1.', 'value2', 'value3', 'value4']}

X_train = train.drop('y', axis=1)
y_train = train['y']
y_train = y_train.replace(('yes', 'no'), (1, -1))

X_test = test.drop('y', axis=1)
y_test = test['y']
y_test = y_test.replace(('yes', 'no'), (1, -1))


initial_weights_value = 1 / len(y_train)
initial_weights = pd.DataFrame({'weights': [initial_weights_value] * len(y_train)})


# Create a decision tree learner
dt = DecisionTree(boosting=True,  T=10, weights=initial_weights)

# Learn a decision tree from the training data
dt.fit(X_train, attributes, y_train)

# Evaluate the decision tree on the testing data
# indicate how many of the generated trees to use with parameter: num_trees
prediction = dt.predict(test_data, attributes, num_trees=10)

# error
error = (prediction.label != y_train).sum() / len(y_train)


```

Using the `DecisionTree` class with random forest:

```python


from decision_tree import DecisionTree

# Load the data from a file
train = pd.read_csv('..train.csv', names=bank_column_names)
test = pd.read_csv('..test.csv', names=bank_column_names)
attributes = {'at1': ['value1'],
    'at2': ['value1.', 'value2', 'value3', 'value4']}

X_train = train.drop('y', axis=1)
y_train = train['y']
y_train = y_train.replace(('yes', 'no'), (1, -1))

X_test = test.drop('y', axis=1)
y_test = test['y']
y_test = y_test.replace(('yes', 'no'), (1, -1))




# Create a decision tree learner
dt = DecisionTree(random_forest=True,  T=10)

# Learn a decision tree from the training data
dt.fit(X_bank_train, bank_attributes, y_bank_train, feature_size=4, replacement=True, sample_frac=0.25)


# Evaluate the decision tree on the testing data
# indicate how many of the generated trees to use with parameter: num_trees
prediction = dt.predict(test_data, attributes, num_trees=10)

# error
error = (prediction.label != y_train).sum() / len(y_train)

```

Using Regression

```python

from linear_regression import LinearRegression
import numpy as np


# Load the data from a file
X_training = np.genfromtxt('../data/train.csv', delimiter=',')
y_training = X_training[:, -1]
X_training = np.insert(X_training, 0, 1, axis=1)
X_training = X_training[:, :-1]

#gradient descent variant can be 'batch', 'stochastic', tol is the tolerance for the stopping condition
regression = LinearRegression(grad_variant='batch', r=0.01, max_iter=float("inf"), tol=1e-6)
regression.fit(X_training, y_training)
prediction = regression.predict(X_training)
weights = regression.get_weights()

```

Using Perceptron

```python

from perceptron import Perceptron
import numpy as np

X_training = np.genfromtxt('data/train.csv', delimiter=',')
X_test = np.genfromtxt('data/test.csv', delimiter=',')
y_training = X_training[:, -1]
y_test = X_test[:, -1]
y_training[y_training == 0] = -1
y_test[y_test == 0] = -1

X_training = np.insert(X_training, 0, 1, axis=1)
X_test = np.insert(X_test, 0, 1, axis=1)

X_training = X_training[:, :-1]
X_test = X_test[:, :-1]

#variant can be 'standard', 'voted', 'average', r is the learning rate, epoch represents the number of iterations
clf = Perceptron(variant='standard', epoch=1, r=0.01)
clf.fit(X_training, y_training)
y_predicted = clf.predict(X_test)


```
