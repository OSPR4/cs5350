This is a machine learning library develop by Osee Pierre for University of Utah's CS5350 course.

    Parameters:
    -----------
    info_gain_variant : str, optional (default='entropy')
        The variant of information gain to use. Can be either 'entropy' or 'gini'.
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
dt = DecisionTree()

# Learn a decision tree from the training data
dt.fit(X_train, attributes, y_train)

# Evaluate the decision tree on the testing data
prediction = dt.predict(test_data, attributes)

# error
error = (prediction.label != y_train).sum() / len(y_train)


```
