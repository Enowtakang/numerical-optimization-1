from numpy.random import rand
from core_functions import make_regression
from core_functions import objective
from core_functions import predict_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from numpy.random import seed
from algorithms import hill_climbing
from algorithms import random_restarts
from algorithms import iterated_local_search
from algorithms import genetic_algorithm
from algorithms import es_comma
from algorithms import es_plus
from algorithms import simulated_annealing


"""
Define Dataset
"""
X, y = make_regression()


"""
Split dataset into train and test sets
"""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=2023)


"""
Define Hyper-parameters
"""
seed(1)     # seed the pseudorandom number generator
n_iter = 10000    # (1000) Total Iterations
step_size = 0.15    # Maximum step size
n_coef = X.shape[1] + 1     # Number of coefficients
solution = rand(n_coef).tolist()    # Define initial solution
n_restarts = 100     # (30) Maximum number of restarts
p_size = 1.0        # Pertubation step size
n_bits = 16     # bits per variable
n_pop = 1000     # define the population size
r_cross = 0.9   # crossover rate
r_mut = 1.0 / (float(n_bits) * len(solution))   # mutation rate
mu = 200     # number of parents selected
lam = 1000   # the number of children generated by parents
temp = 10   # initial temperature


"""
Initialize Algorithms for Algorithmic Search
Computation
"""


def hill_climbing_initialized():
    coefficients, score = hill_climbing(
        X_train, y_train, objective,
        solution, n_iter, step_size)

    return [coefficients, score]


def shc_with_random_restarts_initialized():
    coefficients, score = random_restarts(
        X_train, y_train, objective,
        solution, n_iter, step_size, n_restarts)

    return [coefficients, score]


def iterated_local_search_initialized():
    coefficients, score = iterated_local_search(
        X_train, y_train, objective,
        solution, n_iter, step_size, n_restarts, p_size)

    return [coefficients, score]


def genetic_algorithm_for_cfo_initialized():
    coefficients, score = genetic_algorithm(
        X_train, y_train, objective,
        solution, n_bits, n_iter, n_pop, r_cross, r_mut)

    return [coefficients, score]


def es_coma_initialized():
    coefficients, score = es_comma(
        X_train, y_train, objective,
        solution, n_iter, step_size, mu, lam)

    return [coefficients, score]


def es_plus_initialized():
    coefficients, score = es_plus(
        X_train, y_train, objective,
        solution, n_iter, step_size, mu, lam)

    return [coefficients, score]


def simulated_annealing_initialized():
    coefficients, score = simulated_annealing(
        X_train, y_train, objective,
        solution, n_iter, step_size, temp)

    return [coefficients, score]
