from numpy.random import rand

from core_functions import make_regression
from core_functions import objective
from core_functions import predict_dataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import explained_variance_score as evs
from sklearn.metrics import max_error as max_e
from sklearn.metrics import median_absolute_error as med_ae
from sklearn.metrics import mean_squared_log_error as msle
from sklearn.metrics import mean_gamma_deviance as mgd
from sklearn.metrics import mean_poisson_deviance as mpd
from sklearn.metrics import mean_tweedie_deviance as mtd

from algorithms_initialized import hill_climbing_initialized
from algorithms_initialized import shc_with_random_restarts_initialized
from algorithms_initialized import iterated_local_search_initialized
from algorithms_initialized import genetic_algorithm_for_cfo_initialized
from algorithms_initialized import es_coma_initialized
from algorithms_initialized import es_plus_initialized
from algorithms_initialized import simulated_annealing_initialized


# Define dataset
X, y = make_regression()

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=2023)

"""
Perform Algorithmic Search
"""
coefficients, score = hill_climbing_initialized()


"""
Results
"""

# Prints
print('Done!')
print('Coefficients: %s' % coefficients)
print('Train MSE: %f' % score)

# Generate predictions for the test dataset
yhat = predict_dataset(X_test, coefficients)

# Evaluate Predictions
metrics = [mse, mae, evs, max_e,
           med_ae, msle, mgd, mpd, mtd]

for metric in metrics:
    error_score = round(metric(y_test, yhat), 2)
    print(f"Test__ {metric} __: {error_score}")
