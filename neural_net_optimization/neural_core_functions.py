import numpy
import pandas as pd
from numpy.random import rand, randn
from sklearn.metrics import mean_squared_error as mse


"""
1. Transfer function:
    Rectified Linear Activation Unit (ReLU)
"""


def transfer_relu(activation):
    return max(0.0, activation)


"""
2. Activation function
"""


def activate(row, weights):
    # add the BIAS, the last weight
    activation = weights[-1]

    # add the weight input
    for i in range(len(row)):
        activation += weights[i] * row[i]

    return activation


"""
3. Activation function for a network
"""


def predict_row(row, network):
    inputs = row

    # enumerate the layers in the network from
    # input to output
    for layer in network:
        new_inputs = list()
        # enumerate nodes in the layer
        for node in layer:
            # activate the node
            activation = activate(inputs, node)
            # transfer activation
            output = transfer_relu(activation)
            # store output
            new_inputs.append(output)

        # Output from this layer is input to
        # the next layer
        inputs = new_inputs

    return inputs[0]


"""
4. Use model weights to generate predictions 
for a dataset of rows 
"""


def predict_dataset(X, network):
    yhats = list()

    for row in X:
        yhat = predict_row(row, network)
        yhats.append(yhat)

    return yhats


"""
5. Objective function
"""


def objective(X, y, network):
    # Generate predictions for dataset
    yhat = predict_dataset(X, network)

    # Calculate accuracy
    score = mse(y, yhat)

    return score


"""
6. Step: Take a step in the search space
"""


def step(network, step_size):
    new_net = list()

    # enumerate layers in the network
    for layer in network:
        new_layer = list()

        # enumerate nodes in this layer
        for node in layer:
            # mutate the node
            new_node = node.copy() + randn(len(node)) * step_size
            # store node in layer
            new_layer.append(new_node)

        # store layer in network
        new_net.append(new_layer)

    return new_net


"""
7. Load dataset
"""


def make_regression():
    # Load dataset
    data = pd.read_csv('data.csv')

    # Define X, y
    y = data['ALLSKY_SFC_PAR_TOT']
    X = data.drop('ALLSKY_SFC_PAR_TOT', axis=1)

    # Convert to numpy array
    X = numpy.asarray(X)
    y = numpy.asarray(y)

    # print(X.shape)
    # print(y.shape)

    return [X, y]
