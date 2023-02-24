"""Mean removal on matrices with Python.

This file contains a number of basic examples of how mean removal can be 
implemented using numpy and sklearn packages.

Typical usage example:
    result = mean_removal(matrix)
    diy_result = diy_mean_removal(matrix)
"""

import colorama
import numpy as np
from colorama import Back
from pyfiglet import Figlet
from sklearn import preprocessing


def mean_removal(matrix: np.array) -> np.array:
    """Mean removal with sklearn

    Scales values in a matrix such that the mean of each column is 0 and the
    standard deviation of each column is 1.

    Args:
        matrix (np.array): A numpy array of input values.

    Returns:
        An output array with the transformed elements.
    """
    # The default axis for scaling is column-wise so the mean of each column
    # will be 0, and the standard deviation of each column will be 1
    return preprocessing.scale(matrix)


def diy_mean_removal(matrix: np.array) -> np.array:
    """DIY implementation of mean removal iteratively

    Scales values in a matrix such that the mean of each column is 0 and the
    standard deviation of each column is 1.

    Args:
        matrix (np.array): A numpy array of input values.

    Returns:
        An output array with the transformed elements.
    """
    result = np.empty(matrix.shape)
    col_means = matrix.mean(axis=0)
    col_std_devs = matrix.std(axis=0)
    for row_index, row in enumerate(matrix):
        for col_index, _ in enumerate(row):
            result[row_index][col_index] = (
                matrix[row_index][col_index] - col_means[col_index]
            ) / col_std_devs[col_index]
    return result


def main() -> None:
    """Performs a demonstration of the above functions"""
    colorama.init(autoreset=True)

    figlet = Figlet()
    print(figlet.renderText("Mean Removal"))

    matrix: np.array = np.array(
        [
            [5.1, -2.9, 3.3],
            [-1.2, 7.8, -6.1],
            [3.9, 0.4, 2.1],
            [7.3, -9.9, -4.5],
        ]
    )

    print(Back.YELLOW + "Input matrix:")
    print(f"\n\n{matrix}\n\n")

    result: np.array = mean_removal(matrix)
    diy_result = diy_mean_removal(matrix)

    print(Back.GREEN + "Result using sklearn:")
    print(f"\n\n{result}\n\n")

    print(Back.MAGENTA + "Result using DIY iteration:")
    print(f"\n\n{diy_result}\n\n")


main()
