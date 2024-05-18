import csv
import os
import matplotlib.pyplot as plt

import numpy as np
import time
import pandas as pd

from labwork1.funcs import \
    rastrigin, \
    himmelblau_function, \
    three_hump_camel_function

from labwork1.tracking import \
    differential_evolution__tracking_changes, \
    tabu_search__tracking_changes, \
    mutual_altruism_optimization__tracking_changes

from labwork1.methods import tabu_search, \
    threshold_accepting, \
    deterministic_local_search, \
    differential_evolution, \
    mutual_altruism_optimization, \
    guided_local_search, \
    variable_neighborhood_search

func_list = [rastrigin, himmelblau_function, three_hump_camel_function]

tracking_method_list = [
    tabu_search__tracking_changes,
    mutual_altruism_optimization__tracking_changes,
    differential_evolution__tracking_changes
]
method_list = [
    differential_evolution,
    tabu_search,
    mutual_altruism_optimization,
]
method_list_lab1 = [
    deterministic_local_search,
    variable_neighborhood_search,
    guided_local_search,
    tabu_search,
    threshold_accepting
]
bounds = {
    'rastrigin': [(-5.12, 5.12), (-5.12, 5.12)],  # search domain
    'himmelblau_function': [(-5, 5), (-5, 5)],  # search domain
    'three_hump_camel_function': [(-5, 5), (-5, 5)]  # search domain
}

MAX_ITERATION = 100
ITERATION_ADDED = 1


def plot_fitness_changes(title, df):
    plt.figure(figsize=(10, 5))

    # Plot max_fitness_changes and avg_fitness_changes
    plt.plot(df['iteration'], df['max_fitness_changes'], label='Max Fitness Changes', marker='o', linestyle='-')
    plt.plot(df['iteration'], df['avg_fitness_changes'], label='Avg Fitness Changes', marker='x', linestyle='--')

    # Adding titles and labels
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Changes')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


def plot_fitness_changes_by_iteration(title, df):
    plt.figure(figsize=(10, 5))

    # Plot max_fitness_changes and avg_fitness_changes
    plt.plot(df['iteration'], df['best_fitness'], label='Fitness', linestyle='-')

    # Adding titles and labels
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Fitness Changes')
    plt.legend()
    plt.grid(True)

    # Show plot
    plt.show()


def test(file_path, func, method, iteration=ITERATION_ADDED):
    file_with_dir_path = "./result/" + file_path + ".csv"
    xlsx_dir_path = "./xlsx/" + file_path + ".xlsx"

    os.makedirs(os.path.dirname(file_with_dir_path), exist_ok=True)
    os.makedirs(os.path.dirname(xlsx_dir_path), exist_ok=True)

    with open(file_with_dir_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iteration', 'x', 'y', 'best_fitness', 'execution_time'])

    while iteration <= MAX_ITERATION:
        start_time = time.time()
        match method.__name__:

            case 'differential_evolution' | 'mutual_altruism_optimization' | "simulated_annealing_optimize":
                best_solution, best_fitness = method(func,
                                                     max_iter=iteration,
                                                     bounds=bounds[func.__name__])
            case _:
                best_solution, best_fitness = method(func,
                                                     max_iter=iteration,
                                                     initial_solution=initial_solution)
        end_time = time.time()
        execution_time = end_time - start_time
        with open(file_with_dir_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                iteration,
                round(best_solution[0], 4),
                round(best_solution[1], 4),
                round(best_fitness, 4),
                round(execution_time, 4)
            ])

        if (iteration == MAX_ITERATION):
            print(f"Найкращий розв'язок {func.__name__} знайдено в точці: {best_solution}")
            print(f"Значення функції {func.__name__}  в цій точці: {best_fitness}")
            print()

        iteration += ITERATION_ADDED

    df = pd.read_csv(file_with_dir_path)
    plot_name = func.__name__ + "__" + method.__name__
    plot_fitness_changes_by_iteration(plot_name, df)
    df.to_excel(xlsx_dir_path, index=False)

def test2(file_path, func, method):
    file_with_dir_path = "./result/" + file_path + ".csv"
    xlsx_dir_path = "./xlsx/" + file_path + ".xlsx"

    os.makedirs(os.path.dirname(file_with_dir_path), exist_ok=True)  # Ensure directory for CSV exists
    os.makedirs(os.path.dirname(xlsx_dir_path), exist_ok=True)  # Ensure directory for Excel exists

    with open(file_with_dir_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['iteration', 'max_fitness_changes', 'avg_fitness_changes'])

    best_solution, best_fitness, max_fitness_changes, avg_fitness_changes, iteration = method(func,
                                                                                              bounds=bounds[
                                                                                                  func.__name__])

    print(f"Найкращий розв'язок {func.__name__} знайдено в точці: {best_solution} Кількість ітерацій {iteration}")
    print(f"Значення функції {func.__name__}  в цій точці: {best_fitness}")
    print()

    for i in range(iteration):
        with open(file_with_dir_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i, max_fitness_changes[i - 1], avg_fitness_changes[i - 1]])

    df = pd.read_csv(file_with_dir_path)
    plot_name = func.__name__ + "__" + method.__name__
    plot_fitness_changes(plot_name, df)
    df.to_excel(xlsx_dir_path, index=False)


if __name__ == '__main__':
    # exp_name = "iteration_lab1"
    exp_name = "iteration"
    initial_solution = np.random.uniform(-10, 10, 2)
    print(f"Початкові кординати {initial_solution}")

    for method in method_list_lab1:
        for func in func_list:
            file_path = exp_name + "/"
            file_path += method.__name__ + "__"
            file_path += func.__name__

            print(f"Optimizing function: {func.__name__}")
            print(f"Using method: {method.__name__}")

            iteration = ITERATION_ADDED
            test(file_path, func, method, iteration=iteration)

    # print("--------------------------------------------------------------------------------------------------------")
    # exp_name = "tracking"
    # for method in tracking_method_list:
    #     for func in func_list:
    #         file_path = exp_name + "/"
    #         file_path += method.__name__ + "__"
    #         file_path += func.__name__
    #
    #         print(f"Optimizing function: {func.__name__}")
    #         print(f"Using method: {method.__name__}")
    #
    #         test2(file_path, func, method)
