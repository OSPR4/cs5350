import numpy as np

class Node():
    def __init__(self):
        self.next_layer_nodes = {}
        self.inputs = {}
        self.activation = 0
        self.delta = 0
    
    def add_next_layer_node(self, node, weight):
        self.next_layer_nodes[node] = weight

class NeuralNetwork:
    def __init__(self, features, layers=1, units=5, epochs=10, learning_rate=0.1, tolerance=0.0001, random_weights=True):
        self.layers = layers
        self.features = features
        self.units = units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.random_weights = random_weights
        self.neural_network = {}
        self.build_neural_network()

    def fit(self, X, y):
        error = self.compute_error(X, y)
        convergence = False
        iterations = 0

        while not convergence and iterations < self.epochs:
            shuffle_indices = np.random.permutation(len(X))
            X = X[shuffle_indices]
            y = y[shuffle_indices]
            for j in range(len(X)):
             self.forward_propagation(X[j])
             self.back_propagation_deltas(y[j])
             self.compute_gradients()
             self.update_weights()
            current_error = self.compute_error( X, y)
            error_difference = abs(current_error - error)
            if error_difference < self.tolerance:
                break
            error = current_error
            iterations += 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def build_neural_network(self):
        # layers: number of layers
        # units: number of units in each layer
        neural_network = {}
        layer_node = []
        nn_next_layer_nodes = []

        for i in range(self.layers, -1, -1):
            neural_network[i] = []
            if i == self.layers:
                output_node = Node()
                neural_network[i].append(output_node)
                nn_next_layer_nodes.append(output_node)
            elif i > 0:
                for j in range(self.units + 1):
                    node = Node()
                    neural_network[i].append(node)
                    for next_layer_node in nn_next_layer_nodes:
                        w = 0
                        if self.random_weights:
                            w = np.random.normal(size=1)[0]
                        node.next_layer_nodes[next_layer_node] = (w, 0)
                    if j == 0:
                        node.activation = 1
                        continue
                    layer_node.append(node)
                nn_next_layer_nodes = []
                for node in layer_node:
                    nn_next_layer_nodes.append(node)
                layer_node.clear()
            else:
                for j in range(self.features + 1):
                    node = Node()
                    neural_network[i].append(node)
                    for next_layer_node in nn_next_layer_nodes:
                        w = 0
                        if self.random_weights:
                            w = np.random.normal(size=1)[0]
                        node.next_layer_nodes[next_layer_node] = (w, 0)
                    if j == 0:
                        node.activation = 1
        sorted_nn_dict = {k: neural_network[k] for k in sorted(neural_network)}

        self.neural_network = sorted_nn_dict

        return sorted_nn_dict

    def forward_propagation(self, inputs):
        #append 1 to the beginning of the inputs
        inputs = np.insert(inputs, 0, 1)

        for i in range(len(inputs)):
            self.neural_network[0][i].activation = inputs[i]

        for layer_idx, layer in self.neural_network.items():
            if layer_idx == 0:
                self.input_layer_to_hidden(layer)
            elif layer_idx == len(self.neural_network) - 1:
                self.hidden_layer_to_output(layer)
            else:
                self.hidden_layer_to_hidden(layer)

    def input_layer_to_hidden(self, layer):
        for idx, node in enumerate(layer):
            for nl_node in node.next_layer_nodes:
                input = node.activation
                weight = node.next_layer_nodes[nl_node][0]
                output = input * weight
                nl_node.inputs[idx] = output
                nl_node.activation = output

    def hidden_layer_to_hidden(self, layer):
        for idx, node in enumerate(layer):
            activation = 0
            if idx != 0:
                sigmoid_input = 0
                for i in node.inputs:
                    sigmoid_input += node.inputs[i]
                activation = self.sigmoid(sigmoid_input)
            else:
                activation = 1
            node.activation = activation

            for nl_node in node.next_layer_nodes:
                w = node.next_layer_nodes[nl_node][0]
                output = activation * w
                nl_node.inputs[idx] = output
    
    def hidden_layer_to_output(self, layer):
        node = layer[0]
        output = 0
        for i in node.inputs:
            output += node.inputs[i]
        node.activation = output

    def back_propagation_deltas(self, target):
        for layer_idx in range(len(self.neural_network) - 1, -1, -1):
            layer = self.neural_network[layer_idx]
            if layer_idx == len(self.neural_network) - 1:
                output_node = layer[0]
                output_node.delta = output_node.activation - target
            elif layer_idx == 0:
                pass
            else:
                for node in layer:
                    delta = 0
                    for nl_node in node.next_layer_nodes:
                        nl_node_delta = nl_node.delta
                        w = node.next_layer_nodes[nl_node][0]
                        delta += nl_node_delta * w * node.activation * (1 - node.activation)
                    node.delta = delta

    def compute_gradients(self):
        for layer_idx, layer in self.neural_network.items():
            if layer_idx == len(self.neural_network) - 1:
                continue
            else:
                for idx, node in enumerate(layer):
                    for nl_node in node.next_layer_nodes:
                        nl_node_delta = nl_node.delta
                        partial_derivative = nl_node_delta * node.activation
                        node.next_layer_nodes[nl_node] = (node.next_layer_nodes[nl_node][0], partial_derivative)

    def update_weights(self):
        for layer_idx, layer in self.neural_network.items():
            if layer_idx == len(self.neural_network) - 1:
                continue
            else:
                for node in layer:
                    for nl_node in node.next_layer_nodes:
                        w = node.next_layer_nodes[nl_node][0]
                        partial_derivative = node.next_layer_nodes[nl_node][1]
                        w = w - self.learning_rate * partial_derivative
                        node.next_layer_nodes[nl_node] = (w, partial_derivative)

    def compute_error(self, X, y):
        error = 0
        for i in range(len(X)):
            self.forward_propagation(X[i])
            output_node = self.neural_network[len(self.neural_network) - 1][0]
            error += (output_node.activation - y[i]) ** 2
        return error / len(X)
    
    def predict(self, X):
        predictions = []
        for i in range(len(X)):
            self.forward_propagation(X[i])
            output_node = self.neural_network[len(self.neural_network) - 1][0]
            predictions.append(output_node.activation)
        return predictions
    
