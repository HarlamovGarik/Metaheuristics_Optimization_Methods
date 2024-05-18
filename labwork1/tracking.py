import numpy as np

MAX_ITER = 1000
STAGNATION_THRESHOLD = 500
CONVERGENCE_THRESHOLD = 1e-2

def tabu_search__tracking_changes(
        objective_function,
        bounds,
        tabu_size=10,
        max_iter=MAX_ITER,
):
    bounds = np.array(bounds)
    current_solution = np.random.uniform(bounds[:, 0], bounds[:, 1])
    current_fitness = objective_function(current_solution)
    best_solution = current_solution.copy()
    best_fitness = current_fitness

    max_fitness_changes = []
    avg_fitness_changes = []
    prev_max_fitness = best_fitness
    prev_avg_fitness = current_fitness

    tabu_list = [current_solution.tolist()]  # Initialize tabu list
    iteration = 0

    while iteration < max_iter:
        # Generate a new candidate solution within bounds
        new_solution = current_solution + np.random.normal(0, 0.1, size=current_solution.shape)
        new_solution = np.clip(new_solution, bounds[:, 0], bounds[:, 1])
        new_fitness = objective_function(new_solution)

        # Check if new solution is tabu
        if new_solution.tolist() not in tabu_list:
            if new_fitness < current_fitness:
                # Accept better solutions
                current_solution = new_solution
                current_fitness = new_fitness
                if new_fitness < best_fitness:
                    best_solution = new_solution
                    best_fitness = new_fitness
            else:
                # Use a Metropolis-like criterion to potentially accept worse solutions
                delta_fitness = new_fitness - current_fitness
                acceptance_probability = np.exp(-delta_fitness)
                if np.random.rand() < acceptance_probability:
                    current_solution = new_solution
                    current_fitness = new_fitness

            # Update the tabu list
            tabu_list.append(new_solution.tolist())
            if len(tabu_list) > tabu_size:
                tabu_list.pop(0)  # Maintain the tabu list size


        if iteration > 0 and max_fitness_changes and max_fitness_changes[-1] <= 0.01:
            break

        current_max_fitness = max(current_fitness, best_fitness)
        current_avg_fitness = (current_fitness + best_fitness) / 2

        if prev_max_fitness is not None:
            max_fitness_changes.append(abs(current_max_fitness - prev_max_fitness))
            avg_fitness_changes.append(abs(current_avg_fitness - prev_avg_fitness))

        prev_max_fitness = current_max_fitness
        prev_avg_fitness = current_avg_fitness

        iteration += 1

    return best_solution, best_fitness, max_fitness_changes, avg_fitness_changes, iteration


def differential_evolution__tracking_changes(
        objective_function,
        bounds,
        population_size=20,
        F=0.8,
        CR=0.9,
        max_iter=MAX_ITER,
):
    population = np.random.rand(population_size, len(bounds)) * (np.array([b[1] - b[0] for b in bounds])) + np.array(
        [b[0] for b in bounds])
    fitness = np.array([objective_function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]
    best_solution = population[best_idx]

    max_fitness_changes = []
    avg_fitness_changes = []
    prev_max_fitness = best_fitness
    prev_avg_fitness = np.mean(fitness)

    stagnation_counter = 0  # Counter to track the number of iterations without significant changes
    iteration = 0

    while iteration < max_iter:
        for i in range(population_size):
            indices = [idx for idx in range(population_size) if idx != i]
            a, b, c = population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + F * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])
            cross_points = np.random.rand(len(bounds)) < CR
            trial = np.where(cross_points, mutant, population[i])
            trial_fitness = objective_function(trial)

            if trial_fitness < fitness[i]:
                population[i] = trial
                fitness[i] = trial_fitness
                current_fitness = trial_fitness
                if trial_fitness < best_fitness:
                    best_fitness = trial_fitness
                    best_solution = trial
                    current_fitness = trial_fitness

        if iteration > 0 and max_fitness_changes and max_fitness_changes[-1] <= 0.01:
            break

        current_max_fitness = max(current_fitness, best_fitness)
        current_avg_fitness = (current_fitness + best_fitness) / 2

        if prev_max_fitness is not None:
            max_fitness_changes.append(current_max_fitness - prev_max_fitness)
            avg_fitness_changes.append(current_avg_fitness - prev_avg_fitness)

        prev_max_fitness = current_max_fitness
        prev_avg_fitness = current_avg_fitness

        iteration += 1

    return best_solution, best_fitness, max_fitness_changes, avg_fitness_changes, iteration


def mutual_altruism_optimization__tracking_changes(
        objective_function,
        bounds,
        population_size=50,
        altruism_threshold=0.1,

        max_iter=MAX_ITER,

):
    # Initialize population
    population = np.random.rand(population_size, len(bounds)) * (np.array([b[1] - b[0] for b in bounds])) + np.array(
        [b[0] for b in bounds])
    fitness = np.array([objective_function(ind) for ind in population])
    best_idx = np.argmin(fitness)
    best_fitness = fitness[best_idx]

    prev_max_fitness = best_fitness
    prev_avg_fitness = np.mean(fitness)
    max_fitness_changes = []
    avg_fitness_changes = []

    stagnation_counter = 0  # Counter to monitor stagnation
    iteration = 0

    while iteration < max_iter:
        new_population = population.copy()
        for i in range(population_size):
            if np.random.rand() < altruism_threshold:  # Random chance of altruism
                j = np.random.choice([x for x in range(population_size) if x != i])
                for k in range(len(bounds)):
                    if np.random.rand() < 0.5:  # Randomly choose features to exchange
                        new_population[j][k], new_population[i][k] = new_population[i][k], new_population[j][k]

        new_fitness = np.array([objective_function(ind) for ind in new_population])

        # Check for improvements and update the population
        for i in range(population_size):
            if new_fitness[i] < fitness[i]:
                population[i] = new_population[i]
                fitness[i] = new_fitness[i]
                current_fitness = new_fitness[i]

        if iteration > 0 and max_fitness_changes and max_fitness_changes[-1] <= 0.01:
            break

        current_max_fitness = max(current_fitness, best_fitness)
        current_avg_fitness = (current_fitness + best_fitness) / 2

        if prev_max_fitness is not None:
            max_fitness_changes.append(abs(current_max_fitness - prev_max_fitness))
            avg_fitness_changes.append(abs(current_avg_fitness - prev_avg_fitness))

        prev_max_fitness = current_max_fitness
        prev_avg_fitness = current_avg_fitness

        iteration += 1

    return population[np.argmin(fitness)], np.min(fitness), max_fitness_changes, avg_fitness_changes, iteration