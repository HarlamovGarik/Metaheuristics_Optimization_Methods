import numpy as np


def rastrigin(X):
    return 10 * len(X) + sum([(x ** 2 - 10 * np.cos(2 * np.pi * x)) for x in X])


def himmelblau_function(point):
    return (point[0] ** 2 + point[1] - 11) ** 2 + (point[0] + point[1] ** 2 - 7) ** 2


def three_hump_camel_function(point):
    return 2 * point[0] ** 2 - 1.05 * point[0] ** 4 + (point[0] ** 6 / 6) + point[0] * point[1] + point[1] ** 2


def matsyas_function(point):
    return 0.26 * (point[0] ** 2 + point[1] ** 2) - 0.48 * point[0] * point[1]


def levy_n13_function(point):
    term1 = np.sin(3 * np.pi * point[0]) ** 2
    term2 = (point[0] - 1) ** 2 * (1 + np.sin(3 * np.pi * point[1]) ** 2)
    term3 = (point[1] - 1) ** 2 * (1 + np.sin(2 * np.pi * point[1]) ** 2)
    return term1 + term2 + term3


def schaffer_n4_function(point):
    num = np.cos(np.sin(abs(point[0] ** 2 - point[1] ** 2))) ** 2 - 0.5
    denom = (1 + 0.001 * (point[0] ** 2 + point[1] ** 2)) ** 2
    return 0.5 - num / denom
