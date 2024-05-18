import numpy as np
import pandas as pd
import time

# Алгоритми стохастичного локального пошуку
# Алгоритм імітації відпалу (АІВ)
# Повторюваний локальний пошук
# Алгоритми прискореного ймовірнісного моделювання: g-алгоритми

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 30)
pd.set_option('display.width', None)


# Objective functions
def rastrigin(X):
    return 10 * len(X) + sum([(x ** 2 - 10 * np.cos(2 * np.pi * x)) for x in X])


def himmelblau_function(point):
    return (point[0] ** 2 + point[1] - 11) ** 2 + (point[0] + point[1] ** 2 - 7) ** 2


def three_hump_camel_function(point):
    return 2 * point[0] ** 2 - 1.05 * point[0] ** 4 + (point[0] ** 6 / 6) + point[0] * point[1] + point[1] ** 2


# Stochastic search algorithms
def simulated_annealing(func, initial_temp, cooling_rate, max_iter, dim, domain, step_size):
    current_solution = np.random.uniform(domain[0], domain[1], dim)
    current_value = func(current_solution)
    temp = initial_temp
    best_solution = current_solution
    best_value = current_value

    for i in range(max_iter):
        neighbor = current_solution + np.random.uniform(-step_size, step_size, dim) * (domain[1] - domain[0])
        neighbor_value = func(neighbor)
        delta_value = neighbor_value - current_value
        if delta_value < 0 or np.random.rand() < np.exp(-delta_value / temp):
            current_solution = neighbor
            current_value = neighbor_value
            if neighbor_value < best_value:
                best_solution = neighbor
                best_value = neighbor_value
        temp *= cooling_rate
    return best_solution, best_value


def stochastic_local_search(func, domain, dim, max_iter, step_size):
    best_solution = np.random.uniform(domain[0], domain[1], dim)
    best_value = func(best_solution)

    for _ in range(max_iter):
        candidate_solution = best_solution + np.random.uniform(-step_size, step_size, dim)
        candidate_solution = np.clip(candidate_solution, domain[0], domain[1])
        candidate_value = func(candidate_solution)

        if candidate_value < best_value:
            best_solution = candidate_solution
            best_value = candidate_value

    return best_solution, best_value


def iterative_local_search(func, max_iter, domain, dim, step_size):
    best_solution = np.random.uniform(domain[0], domain[1], dim)
    best_value = func(best_solution)

    for _ in range(max_iter):
        neighbors = best_solution + np.random.uniform(-step_size, step_size, dim)  # Генерація сусідів
        neighbors = np.clip(neighbors, domain[0], domain[1])  # Збереження рішень в межах домену
        value = func(neighbors)

        if value < best_value:
            best_solution, best_value = neighbors, value
            
    return best_solution, best_value


def random_search(func, max_iter, dim, domain):
    best_solution = np.random.uniform(domain[0], domain[1], dim)
    best_value = func(best_solution)

    for _ in range(max_iter):
        candidate_solution = np.random.uniform(domain[0], domain[1], dim)
        candidate_value = func(candidate_solution)
        if candidate_value < best_value:
            best_solution = candidate_solution
            best_value = candidate_value
    return best_solution, best_value


def g_algorithm(func, population_size, generations, dim, domain, step_size):
    population = np.random.uniform(domain[0], domain[1], (population_size, dim))
    for _ in range(generations):
        fitness = np.array([func(ind) for ind in population])
        best_individuals = population[np.argsort(fitness)[:population_size // 2]]
        offspring = np.array(
            [np.random.uniform(-step_size, step_size, dim) + individual for individual in best_individuals])
        population = np.vstack((best_individuals, offspring))
    best_solution = population[np.argmin([func(ind) for ind in population])]
    best_value = func(best_solution)
    return best_solution, best_value


# Main execution to run tests
def run_tests():
    results = []
    results1 = []
    results2 = []
    results3 = []

    test_functions = [rastrigin, himmelblau_function, three_hump_camel_function]

    for func in test_functions:
        for i in range(200):  # Runs 4 variations per function
            initial_temp = np.random.uniform(5, 20)
            cooling_rate = np.random.uniform(0.1, 0.99)
            max_iter = np.random.choice([10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
            dim = np.random.choice([2])
            domain = np.random.uniform(-10, 0, 1).tolist() + np.random.uniform(0, 10, 1).tolist()
            population_size = np.random.choice([10, 20, 30, 40, 50])
            step_size = np.random.choice(np.arange(0.1, 10.1, 0.1))
            generations = max_iter // 10

            # Simulated Annealing
            start_time = time.time()
            best_solution, best_value = simulated_annealing(func, initial_temp, cooling_rate, max_iter, dim, domain,
                                                            step_size=step_size)
            end_time = time.time()
            results.append({
                'Function': func.__name__,
                'Iterations': max_iter,
                'Best Solution': [round(x, 4) for x in best_solution],
                'Best Fitness': round(best_value, 4),
                'Execution Time (s)': round(end_time - start_time, 4),
                'Initial Temp': round(initial_temp, 4),
                'Cooling Rate': round(cooling_rate, 4),
                'Step Size': round(step_size, 4),
                'Domain': [round(x, 4) for x in domain]
            })

            # Stochastic local search
            start_time = time.time()
            best_solution, best_value = stochastic_local_search(func, domain, dim, max_iter, step_size=step_size)
            end_time = time.time()
            results1.append({
                'Function': func.__name__,
                'Iterations': max_iter,
                'Best Solution': [round(x, 4) for x in best_solution],
                'Best Fitness': round(best_value, 4),
                'Execution Time (s)': round(end_time - start_time, 4),
                'Step Size': round(step_size, 4),
                'Domain': [round(x, 4) for x in domain]
            })

            # Iterative Local Search
            start_time = time.time()
            best_solution, best_value = iterative_local_search(func, max_iter=max_iter, domain=domain,
                                                               step_size=step_size, dim=dim)
            end_time = time.time()
            results2.append({
                'Iterations': max_iter,
                'Function': func.__name__,
                'Best Solution': [round(x, 4) for x in best_solution],
                'Best Fitness': round(best_value, 4),
                'Execution Time (s)': round(end_time - start_time, 4),
                'Step Size': round(step_size, 4),
                'Domain': [round(x, 4) for x in domain]
            })

            # G-Algorithm
            start_time = time.time()
            best_solution, best_value = g_algorithm(func, population_size, generations, dim, domain,
                                                    step_size=step_size)
            end_time = time.time()
            results3.append({
                'Function': func.__name__,
                'Iterations': max_iter,
                'Best Solution': [round(x, 4) for x in best_solution],
                'Best Fitness': round(best_value, 4),
                'Execution Time (s)': round(end_time - start_time, 4),
                'Population Size': population_size,
                'Generations': generations,
                'Step Size': round(step_size, 4),
                'Domain': [round(x, 4) for x in domain]
            })

    df_results1 = pd.DataFrame(results)
    df_results2 = pd.DataFrame(results1)
    df_results3 = pd.DataFrame(results2)
    df_results4 = pd.DataFrame(results3)

    return df_results1, df_results2, df_results3, df_results4


# Displaying sorted results
df1, df2, df3, df4 = run_tests()
df2.to_csv("./stochastic_local_search.csv", index=False)
df3.to_csv("./iterative_local_search.csv", index=False)
df4.to_csv("./g-algorithm.csv", index=False)
print(df3)
