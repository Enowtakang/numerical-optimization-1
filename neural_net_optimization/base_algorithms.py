from numpy.random import rand, randn
from numpy.random import randint
from numpy import argsort
from numpy import exp
from neural_core_functions import step


"""
Algorithm 1: Stochastic hill climbing
"""


def hill_climbing(
        X, y, objective, solution, n_iter, step_size):

    # Evaluate the initial point
    solution_eval = objective(X, y, solution)

    # Run the hill climb
    for i in range(n_iter):
        # Take a step
        candidate = step(solution, step_size)

        # Evaluate the candidate point
        candidate_eval = objective(X, y, candidate)

        # Check if the new point should be kept
        if candidate_eval <= solution_eval:
            # Store the new point
            solution, solution_eval = candidate, candidate_eval

            # Report progress
            print('>%d %.5f' % (i, solution_eval))

    return [solution, solution_eval]


"""
Algorithm 2: Stochastic hill climbing 
with random restarts
"""


def hill_climbing_with_random_restarts(
        X, y, objective, solution,
        n_iter, step_size, start_pt):

    # store the initial point
    solution = start_pt

    # Evaluate the initial point
    solution_eval = objective(X, y, solution)

    # Run the hill climb
    for i in range(n_iter):
        # Take a step
        candidate = step(solution, step_size)

        # Evaluate the candidate point
        candidate_eval = objective(X, y, candidate)

        # Check if the new point should be kept
        if candidate_eval <= solution_eval:
            # Store the new point
            solution, solution_eval = candidate, candidate_eval

    return [solution, solution_eval]


def random_restarts(X, y, objective, solution, n_iter,
                    step_size, n_restarts):

    best, best_eval = None, 1e+10

    # Enumerate restarts
    for n in range(n_restarts):
        start_pt = rand(len(solution))

        # Perform a stochastic hill climbing search
        solution, solution_eval = hill_climbing_with_random_restarts(
            X, y, objective, solution, n_iter, step_size,
            start_pt)

        # Check for new best
        if solution_eval < best_eval:
            best, best_eval = solution, solution_eval
            print('Restart %d %.5f' % (
                n, best_eval))

    return [best, best_eval]


"""
Algorithm 3: Iterated Local Search Algorithm
"""


def hill_climbing_iterated_local_search(
        X, y, objective, solution, n_iter,
        step_size, start_pt):

    # Store the initial point
    solution = start_pt

    # Evaluate the initial point
    solution_eval = objective(X, y, solution)

    # Run the hill climb
    for i in range(n_iter):
        candidate = step(solution, step_size)

        # Evaluate the candidate point
        candidate_eval = objective(X, y, candidate)

        # Check if the new point should be kept
        if candidate_eval <= solution_eval:
            # Store the new point
            solution, solution_eval = candidate, candidate_eval

    return [solution, solution_eval]


def iterated_local_search(
        X, y, objective, solution,
        n_iter, step_size, n_restarts, p_size):

    best = rand(len(solution))

    # Evaluate the current best point
    best_eval = objective(X, y, best)

    # Enumerate restarts
    for i in range(n_restarts):
        start_pt = best + randn(len(solution)) * p_size

        # Perform a stochastic hill climbing search
        solution, solution_eval = hill_climbing_iterated_local_search(
            X, y, objective, solution, n_iter, step_size,
            start_pt)

        # Check for new best
        if solution_eval <= best_eval:
            best, best_eval = solution, solution_eval

            print('Restart %d %.5f' % (
                i, best_eval))

    return [best, best_eval]


"""
Algorithm 4: Genetic Algorithm for 
Continuous function iteration
"""


# decode bitstring to numbers


def decode(solution, n_bits, bitstring):
    decoded = list()
    largest = 2**n_bits
    for i in range(len(solution)):
        # extract the substring
        start, end = i * n_bits, (i * n_bits)+n_bits
        substring = bitstring[start:end]
        # convert bitstring to a string of chars
        chars = ''.join([str(s) for s in substring])
        # convert string to integer
        integer = int(chars, 2)
        # scale integer to desired range
        value = (integer/largest)
        # store
        decoded.append(value)
    return decoded


# tournament selection


def selection(pop, scores, k=3):
    # first random selection
    selection_ix = randint(len(pop))
    for ix in randint(0, len(pop), k-1):
        # check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# crossover two parents to create two children


def crossover(p1, p2, r_cross):
    # children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    # check for recombination
    if rand() < r_cross:
        # select crossover point that is not on the end of the string
        pt = randint(1, len(p1)-2)
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# mutation operator


def mutation(bitstring, r_mut):
    for i in range(len(bitstring)):
        # check for a mutation
        if rand() < r_mut:
            # flip the bit
            bitstring[i] = 1 - bitstring[i]


# genetic algorithm


def genetic_algorithm(
        X, y, objective, solution,
        n_bits, n_iter, n_pop,
        r_cross, r_mut):
    # initial population of random bitstring
    pop = [randint(0, 2, n_bits*len(solution)).tolist() for _ in range(n_pop)]
    # keep track of best solution
    best, best_eval = rand(len(solution)), objective(X, y, decode(solution, n_bits, pop[0]))
    # enumerate generations
    for gen in range(n_iter):
        # decode population
        decoded = [decode(solution, n_bits, p) for p in pop]
        # evaluate all candidates in the population
        scores = [objective(X, y, d) for d in decoded]
        # check for new best solution
        for i in range(n_pop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
        # select parents
        selected = [selection(pop, scores) for _ in range(n_pop)]
        # create the next generation
        children = list()
        for i in range(0, n_pop, 2):
            # get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            # crossover and mutation
            for c in crossover(p1, p2, r_cross):
                # mutation
                mutation(c, r_mut)
                # store for next generation
                children.append(c)
        # replace population
        pop = children
    return [best, best_eval]


"""
Algorithm 5: Evolution Strategy (mu, lambda)
"""


def es_comma(
        X, y, objective, solution,
        n_iter, step_size, mu, lam):
    best, best_eval = rand(len(solution)), 1e+10   # None
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = list()
    for _ in range(lam):
        candidate = step(solution, step_size)
        population.append(candidate)
    # perform the search
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(X, y, c) for c in population]
        # rank scores in ascending order
        ranks = argsort(argsort(scores))
        # select the indexes for the top mu ranked solutions
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        # create children from parents
        children = list()
        for i in selected:
            # check if this parent is the best solution ever seen
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
            # create children for parent
            for _ in range(n_children):
                child = population[i] + randn(len(solution)) * step_size
                children.append(child)
        # replace population with children
        population = children
    return [best, best_eval]


"""
Algorithm 6: Evolution Strategy (mu + lambda)
"""


def es_plus(
        X, y, objective, solution, n_iter,
        step_size, mu, lam):

    best, best_eval = rand(len(solution)), 1e+10   # None
    # calculate the number of children per parent
    n_children = int(lam / mu)
    # initial population
    population = list()
    for _ in range(lam):
        candidate = step(solution, step_size)
        population.append(candidate)
    # perform the search
    for epoch in range(n_iter):
        # evaluate fitness for the population
        scores = [objective(X, y, c) for c in population]
        # rank scores in ascending order
        ranks = argsort(argsort(scores))
        # select the indexes for the top mu ranked solutions
        selected = [i for i, _ in enumerate(ranks) if ranks[i] < mu]
        # create children from parents
        children = list()
        for i in selected:
            # check if this parent is the best solution ever seen
            if scores[i] < best_eval:
                best, best_eval = population[i], scores[i]
                print('%d, Best: f(%s) = %.5f' % (epoch, best, best_eval))
            # keep the parent
            children.append(population[i])
            # create children for parent
            for _ in range(n_children):
                child = population[i] + randn(len(solution)) * step_size
                children.append(child)
        # replace population with children
        population = children
    return [best, best_eval]


"""
Algorithm 7: Simulated Annealing
"""


def simulated_annealing(
        X, y, objective, solution,
        n_iter, step_size, temp):

    # generate an initial point
    best = rand(len(solution))

    # evaluate the initial point
    best_eval = objective(X, y, best)

    # current working solution
    curr, curr_eval = best, best_eval

    # run the algorithm
    for i in range(n_iter):
        # take a step
        candidate = step(solution, step_size)
        # evaluate candidate point
        candidate_eval = objective(X, y, candidate)

        # check for new best solution
        if candidate_eval < best_eval:
            # store new best point
            best, best_eval = candidate, candidate_eval
            # report progress
            print('>%d f(%s) = %.5f' % (i, best, best_eval))

        # difference between candidate and current
        # point evaluation
        diff = candidate_eval - curr_eval
        # calculate temperature for current epoch
        t = temp / float(i + 1)
        # calculate metropolis acceptance criterion
        metropolis = exp(-diff / t)
        # check if we should keep the new point
        if diff < 0 or rand() < metropolis:
            # store the new current point
            curr, curr_eval = candidate, candidate_eval
    return [best, best_eval]
