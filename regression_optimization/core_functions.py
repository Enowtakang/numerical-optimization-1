import numpy
import pandas as pd
from numpy.random import rand, randn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse


"""
Linear Regression
"""


def predict_row(row, coefficients):
    # Add the bias, the last coefficient
    result = coefficients[-1]

    # Add the weighted input
    for i in range(len(row)):
        result += coefficients[i] * row[i]

    return result


"""
Use model coefficients to generate predictions
for a dataset of rows
"""


def predict_dataset(X, coefficients):
    yhats = list()

    for row in X:
        # Make a prediction
        yhat = predict_row(row, coefficients)

        # Store the prediction
        yhats.append(yhat)

    return yhats


"""
Objective function
"""


def objective(X, y, coefficients):
    # Generate predictions for dataset
    yhat = predict_dataset(X, coefficients)

    # Calculate accuracy
    score = mse(y, yhat)

    return score


"""
Load dataset
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
