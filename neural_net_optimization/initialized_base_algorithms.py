from numpy.random import rand, seed
from neural_core_functions import make_regression, objective
from sklearn.model_selection import train_test_split
from base_algorithms import hill_climbing
from base_algorithms import random_restarts
from base_algorithms import iterated_local_search
from base_algorithms import genetic_algorithm
from base_algorithms import es_comma
from base_algorithms import es_plus
from base_algorithms import simulated_annealing


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

# one hidden layer and one output layer
n_inputs = X.shape[1]     # Number of inputs
n_hidden = 10
hidden_1 = [rand(n_inputs + 1) for _ in range(n_hidden)]
output_1 = [rand(n_hidden + 1)]
network = [hidden_1, output_1]

n_iter = 10    # (2000) Total Iterations
n_restarts = 10     # (60) Maximum number of restarts

step_size = 0.15    # Maximum step size
p_size = 1.0        # Pertubation step size
n_bits = 16     # bits per variable
n_pop = 100     # define the population size
r_cross = 0.9   # crossover rate
r_mut = 1.0 / (float(n_bits) * len(network))   # mutation rate
mu = 20     # number of parents selected
lam = 100   # the number of children generated by parents
temp = 10   # initial temperature


"""
Initialize Algorithms for Algorithmic Search
Computation
"""


def hill_climbing_initialized():
    coefficients, score = hill_climbing(
        X_train, y_train, objective,
        network, n_iter, step_size)

    return [coefficients, score]


def shc_with_random_restarts_initialized():
    coefficients, score = random_restarts(
        X_train, y_train, objective,
        network, n_iter, step_size, n_restarts)

    return [coefficients, score]


def iterated_local_search_initialized():
    coefficients, score = iterated_local_search(
        X_train, y_train, objective,
        network, n_iter, step_size, n_restarts, p_size)

    return [coefficients, score]


def genetic_algorithm_for_cfo_initialized():
    coefficients, score = genetic_algorithm(
        X_train, y_train, objective,
        network, n_bits, n_iter, n_pop, r_cross, r_mut)

    return [coefficients, score]


def es_coma_initialized():
    coefficients, score = es_comma(
        X_train, y_train, objective,
        network, n_iter, step_size, mu, lam)

    return [coefficients, score]


def es_plus_initialized():
    coefficients, score = es_plus(
        X_train, y_train, objective,
        network, n_iter, step_size, mu, lam)

    return [coefficients, score]


def simulated_annealing_initialized():
    coefficients, score = simulated_annealing(
        X_train, y_train, objective,
        network, n_iter, step_size, temp)

    return [coefficients, score]