import numpy as np


def feature_penalty_function(solution, penalties, adaptivity_rate=0.1):
    penalty_update = np.abs(solution) / (np.max(np.abs(solution)) + 1e-8)  # Adding epsilon to avoid division by zero
    new_penalties = penalties + adaptivity_rate * penalty_update
    return new_penalties


def tabu_search(
        objective_function,
        initial_solution=None,
        tabu_size=10,
        max_iter=1000,
):
    if initial_solution is None:
        initial_solution = np.random.uniform(-10, 10, 2)

    current_solution = np.array(initial_solution)
    current_fitness = objective_function(current_solution)
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    tabu_list = []

    for iteration in range(max_iter):
        new_solution = current_solution + np.random.normal(0, 0.1, size=current_solution.shape)

        new_fitness = objective_function(new_solution)

        if new_solution.tolist() in tabu_list:
            continue

        if new_fitness < current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness

            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness

            tabu_list.append(new_solution.tolist())
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)

    return best_solution, best_fitness


def differential_evolution(
        objective_function,
        bounds,
        population_size=20,
        F=0.8,
        CR=0.9,
        max_iter=1000,
):
    """
    Differential Evolution optimization algorithm.

    Parameters:
    - objective_function: The function to minimize.
    - bounds: A list of tuples defining the lower and upper bounds for each dimension.
    - population_size: The number of solutions in the population.
    - F: The differential weight used in mutation.
    - CR: The crossover probability.
    - max_iter: The maximum number of iterations to run.
    - convergence_threshold: The fitness improvement threshold to consider convergence.
    - stagnation_threshold: The maximum allowed iterations without improvement.

    Returns:
    - best_solution: The best solution found.
    - best_fitness: The fitness of the best solution.
    - iteration:
    - result: Classification of the optimization result as 'good' or 'bad'.
    """

    population = np.random.rand(population_size, len(bounds)) * (np.array(bounds).ptp(axis=1)) + np.array(bounds)[:, 0]
    best_idx = np.argmin([objective_function(ind) for ind in population])
    best_solution = population[best_idx]
    best_fitness = objective_function(best_solution)

    for iteration in range(max_iter):
        for i in range(population_size):
            indices = [idx for idx in range(population_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])
            cross_points = np.random.rand(len(bounds)) < CR
            trial = np.where(cross_points, mutant, population[i])
            trial_fitness = objective_function(trial)

            if trial_fitness < objective_function(population[i]):
                population[i] = trial
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial

    return best_solution, best_fitness


def mutual_altruism_optimization(
        objective_function,
        bounds,
        population_size=50,
        altruism_threshold=0.1,
        max_iter=1000,
):
    # Initialize population
    population = np.random.rand(population_size, len(bounds)) * (np.array([b[1] - b[0] for b in bounds])) + np.array(
        [b[0] for b in bounds])
    fitness = np.array([objective_function(ind) for ind in population])

    for _ in range(max_iter):
        new_population = population.copy()
        for i in range(population_size):
            if np.random.rand() < altruism_threshold:  # Random chance of altruism
                j = np.random.choice([x for x in range(population_size) if x != i])
                for k in range(len(bounds)):
                    if np.random.rand() < 0.5:  # Randomly choose features to exchange
                        new_population[j][k], new_population[i][k] = new_population[i][k], new_population[j][k]

        new_fitness = np.array([objective_function(ind) for ind in new_population])

        for i in range(population_size):
            if new_fitness[i] < fitness[i]:
                population[i] = new_population[i]
                fitness[i] = new_fitness[i]

    return population[np.argmin(fitness)], np.min(fitness)


def variable_neighborhood_search(
        objective_function,
        initial_solution,
        max_iter=1000,
        initial_radius=1.0,
        shrink_factor=0.9,
        neighborhood_change_trigger=100
):
    """
    Perform Variable Neighborhood Search (VNS) to minimize a given objective function.

    Parameters:
    - objective_function: The function to be minimized.
    - initial_solution: The starting point of the search.
    - max_iter: Maximum number of iterations to perform.
    - initial_radius: Initial radius size for the neighborhood.
    - shrink_factor: Factor by which the neighborhood radius is reduced upon a successful improvement.
    - neighborhood_change_trigger: Iterations after which to change the neighborhood if no improvement.

    Returns:
    - best_solution: The best solution found.
    - best_fitness: The best objective function value found.
    """
    current_solution = np.array(initial_solution)
    best_solution = current_solution.copy()
    best_fitness = objective_function(best_solution)
    current_radius = initial_radius
    iter_since_last_improvement = 0

    for _ in range(max_iter):
        trial_solution = current_solution + np.random.uniform(-current_radius, current_radius,
                                                              size=current_solution.shape)
        trial_fitness = objective_function(trial_solution)

        # Check if the new solution is better
        if trial_fitness < best_fitness:
            best_solution = trial_solution
            best_fitness = trial_fitness
            current_solution = trial_solution
            iter_since_last_improvement = 0  # Reset the counter
            current_radius = initial_radius  # Reset the radius to initial
        else:
            iter_since_last_improvement += 1

        # Increase the neighborhood size if no improvement is found in a defined number of iterations
        if iter_since_last_improvement >= neighborhood_change_trigger:
            current_radius *= (1 / shrink_factor)  # Expand the neighborhood
            iter_since_last_improvement = 0  # Reset the counter

        # Shrink the neighborhood upon each iteration to focus the search
        current_radius *= shrink_factor

    return best_solution, best_fitness


def deterministic_local_search(
        objective_function,
        initial_solution=None,
        step_size=0.1,
        max_iter=1000,
        dimensionality=2,
        step_size_increase_factor=1.5,
        no_improve_limit=20
):
    """
    Perform Deterministic Local Search (DLS) with dynamic step size to minimize a given objective function.

    Parameters:
    - objective_function: The function to be minimized.
    - initial_solution: The starting point of the search.
    - step_size: The initial step size used to explore the neighborhood.
    - max_iter: Maximum number of iterations to perform.
    - dimensionality: The number of dimensions of the problem space.
    - step_size_increase_factor: Factor by which to increase the step size when stuck.
    - no_improve_limit: Number of iterations without improvement before increasing the step size.

    Returns:
    - best_solution: The best solution found.
    - best_fitness: The best objective function value found.
    """
    if initial_solution is None:
        initial_solution = np.random.uniform(-10, 10, 2)

    current_solution = np.array(initial_solution)
    best_solution = current_solution.copy()
    best_fitness = objective_function(best_solution)
    no_improve_count = 0

    for _ in range(max_iter):
        improved = False
        # Generate all neighboring solutions within one step size in each dimension
        for i in range(dimensionality):
            for delta in [-step_size, step_size]:
                trial_solution = best_solution.copy()
                trial_solution[i] += delta
                trial_fitness = objective_function(trial_solution)

                # Update the best found solution if the new one is better
                if trial_fitness < best_fitness:
                    best_solution = trial_solution
                    best_fitness = trial_fitness
                    improved = True
                    no_improve_count = 0  # Reset the counter on improvement

        # Break the loop if no improvement was found in the last iteration
        if not improved:
            no_improve_count += 1
            if no_improve_count >= no_improve_limit:
                step_size *= step_size_increase_factor  # Increase the step size to escape local minima
                no_improve_count = 0  # Reset the counter

    return best_solution, best_fitness


def guided_local_search(
        objective_function,
        initial_solution=None,
        max_iter=1000,
        neighbourhood_size=0.1,
        adaptivity_rate=0.1
):
    if initial_solution is None:
        initial_solution = np.random.uniform(-10, 10, 2)

    current_solution = initial_solution
    current_fitness = objective_function(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness
    penalties = np.zeros(current_solution.shape)

    for _ in range(max_iter):
        new_solution = current_solution + np.random.uniform(-neighbourhood_size, neighbourhood_size,
                                                            size=current_solution.shape)

        new_fitness = objective_function(new_solution)
        penalties = feature_penalty_function(new_solution, penalties)
        new_fitness += np.sum(penalties)

        if new_fitness < current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness

            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
        else:
            neighbourhood_size *= (1 - adaptivity_rate)

    return best_solution, best_fitness


def threshold_accepting(
        objective_function,
        initial_solution=None,
        max_iter=1000,
        threshold=0.1
):
    if initial_solution is None:
        initial_solution = np.random.uniform(-10, 10, 2)

    current_solution = initial_solution
    current_fitness = objective_function(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness

    for _ in range(max_iter):
        new_solution = current_solution + np.random.uniform(-threshold, threshold, size=current_solution.shape)

        new_fitness = objective_function(new_solution)

        if new_fitness < current_fitness:
            current_solution = new_solution
            current_fitness = new_fitness

            if new_fitness < best_fitness:
                best_solution = new_solution
                best_fitness = new_fitness
        else:
            threshold *= 0.9

    return best_solution, best_fitness