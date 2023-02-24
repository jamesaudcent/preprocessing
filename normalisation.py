"""Normalising matrices with Python.

This file contains a number of basic examples of how normalisation can be 
implemented using numpy and sklearn packages.

Typical usage example:
"""

import math

import colorama
import numpy as np
from colorama import Back
from pyfiglet import Figlet
from sklearn import preprocessing


def l1_normalize(matrix: np.array) -> np.array:
    """L1 normalisation with sklearn

    Scale values in a matrix such that the total of a row adds to a value of 1.

    Args:
        matrix (np.array): A numpy array of input values.

    Returns:
        An output array with the transformed elements.
    """
    return preprocessing.normalize(matrix, norm="l1")


def l2_normalize(matrix: np.array) -> np.array:
    """L2 normalisation with sklearn

    Scale values in a matrix such that the total of the squares of the values in
    a single row adds to a value of 1.

    Args:
        matrix (np.array): A numpy array of input values.

    Returns:
        An output array with the transformed elements.
    """
    return preprocessing.normalize(matrix, norm="l2")


def diy_l1_normalize(matrix: np.array) -> np.array:
    """DIY implementation of l1 normalisation

    Scale values in a matrix such that the total of a row adds to a value of 1.

    Args:
        matrix (np.array): A numpy array of input values.

    Returns:
        An output array with the transformed elements.
    """
    result = np.empty(matrix.shape)
    row_abs_sums = np.sum(np.absolute(matrix), axis=1)
    for row_index, row in enumerate(matrix):
        row_sum = row_abs_sums[row_index]
        for col_index, _ in enumerate(row):
            result[row_index][col_index] = (
                matrix[row_index][col_index] / row_sum
            )
    return result


def diy_l2_normalize(matrix: np.array) -> np.array:
    """DIY implementation of l2 normalisation

    Scale values in a matrix such that the total of the squares of the values in
    a single row adds to a value of 1.

    Args:
        matrix (np.array): A numpy array of input values.

    Returns:
        An output array with the transformed elements.
    """
    result = np.empty(matrix.shape)
    row_sqr_sums = np.sum(np.square(matrix), axis=1)
    for row_index, row in enumerate(matrix):
        row_sum = row_sqr_sums[row_index]
        for col_index, _ in enumerate(row):
            result[row_index][col_index] = matrix[row_index][
                col_index
            ] / math.sqrt(row_sum)
    return result


def main() -> None:
    """Performs a demonstration of the above functions"""
    colorama.init(autoreset=True)

    figlet = Figlet()
    print(figlet.renderText("Normalisation"))

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

    l1_result: np.array = l1_normalize(matrix)
    diy_l1_result: np.array = diy_l1_normalize(matrix)

    print(Back.GREEN + "L1 normalized using sklearn:")
    print(f"\n\n{l1_result}\n\n")

    print(Back.MAGENTA + "L1 normalized using DIY iteration:")
    print(f"\n\n{diy_l1_result}\n\n")

    l2_result: np.array = l2_normalize(matrix)
    diy_l2_result: np.array = diy_l2_normalize(matrix)

    print(Back.GREEN + "L2 normalized using sklearn:")
    print(f"\n\n{l2_result}\n\n")

    print(Back.MAGENTA + "L2 normalized using DIY iteration:")
    print(f"\n\n{diy_l2_result}\n\n")


main()
