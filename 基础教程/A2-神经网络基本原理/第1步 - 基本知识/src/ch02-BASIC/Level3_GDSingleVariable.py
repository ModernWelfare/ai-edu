# Copyright (c) Microsoft. All rights reserved.
# Licensed under the MIT license. See LICENSE file in the project root for full license information.

import numpy as np
import matplotlib.pyplot as plt

def target_function(x: float) -> float:
    """
    Returns the value of the quadratic function x^2 at the given input.

    Args:
        x (float): The input to the quadratic function.

    Returns:
        float: The value of the quadratic function at the given input.
    """
    y = x ** 2
    return y

def derivative_function(x: float) -> float:
    """
    Returns the derivative of the quadratic function x^2 at the given input.

    Args:
        x (float): The input to the quadratic function.

    Returns:
        float: The value of the derivative of the quadratic function at the given input.
    """
    dy_dx = 2 * x
    return dy_dx

def plot_function() -> None:
    """
    Plots the quadratic function x^2 in the range [-1.2, 1.2].
    """
    x = np.linspace(-1.2, 1.2)
    y = target_function(x)
    plt.plot(x, y)

def plot_gradient_descent(X: list[float]) -> None:
    """
    Plots the gradient descent path of the quadratic function x^2, given a list of X values.

    Args:
        X (list[float]): A list of x values representing the path of the gradient descent.
    """
    Y = [target_function(x) for x in X]
    plt.plot(X, Y)

if __name__ == '__main__':
    x = 1.2
    eta = 0.4
    error = 1e-3
    X = [x]

    y = target_function(x)
    while y > error:
        x -= eta * derivative_function(x)
        X.append(x)
        y = target_function(x)
        print(f"x={x:.3f}, y={y:.3f}")

    plot_function()
    plot_gradient_descent(X)
    plt.show()
