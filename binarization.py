"""Binarization of matrices with Python.

This file contains a number of basic examples of how binarization can be 
implemented using numpy and sklearn packages.

Typical usage example:
    binarized_data = binarize(matrix, 2.1)
    iterative_binarized_data = iterative_binarize(matrix, 2.1)
    vector_binarized_data = vector_binarize(matrix, 2.1)
"""

import colorama
import numpy as np
from colorama import Back
from pyfiglet import Figlet
from sklearn import preprocessing


def binarize(matrix: np.array, threshold: float) -> np.array:
    """Binarize with sklearn

    Converts numerical elements of an array to boolean based on a theshold
    value, by calling the sklearn preprocessing binarize function.

    Args:
        matrix (np.array): A numpy array of input values.
        threshold (float): The threshold for binarization.

    Returns:
        An output array with binarized elements.
    """
    return preprocessing.Binarizer(threshold=threshold).transform(matrix)


def iterative_binarize(matrix: np.array, threshold: float) -> np.array:
    """DIY implementation of binarize using iteration

    Converts numerical elements of an array to boolean based on a theshold
    value, by iterating over the entirety of the numpy array.

    Args:
        matrix (np.array): A numpy array of input values.
        threshold (float): The threshold for binarization.

    Returns:
        An output array with binarized elements.
    """
    result = np.empty(matrix.shape)
    for row_index, row in enumerate(matrix):
        for col_index, _ in enumerate(row):
            if matrix[row_index][col_index] > threshold:
                result[row_index][col_index] = 1
            else:
                result[row_index][col_index] = 0
    return result


def greater_than_threshold(element: float, threshold: float) -> float:
    """Returns 1 if greater than threshold, else 0"""
    return float(element > threshold)


def vector_binarize(matrix: np.array, threshold: float) -> np.array:
    """DIY implementation of binarize using vectorize

    Converts numerical elements of an array to boolean based on a theshold
    value, by creating a vectorized function.

    Args:
        matrix (np.array): A numpy array of input values.
        threshold (float): The threshold for binarization.

    Returns:
        An output array with binarized elements.
    """
    vectorized_binarizer = np.vectorize(greater_than_threshold)
    return vectorized_binarizer(matrix, threshold)


def main() -> None:
    """Performs a demonstration of the above functions"""
    colorama.init(autoreset=True)

    figlet = Figlet()
    print(figlet.renderText("Binarization"))

    data: np.array = np.array(
        [
            [5.1, -2.9, 3.3],
            [-1.2, 7.8, -6.1],
            [3.9, 0.4, 2.1],
            [7.3, -9.9, -4.5],
        ]
    )

    print(Back.YELLOW + "Input matrix:")
    print(f"\n\n{data}\n\n")

    binarized_data: np.array = binarize(data, 2.1)
    iteratively_binarized_data = iterative_binarize(data, 2.1)
    vector_binarized_data = vector_binarize(data, 2.1)

    print(Back.GREEN + "Result using sklearn binarizer:")
    print(f"\n\n{binarized_data}\n\n")

    print(Back.MAGENTA + "Result using iterative binarizer:")
    print(f"\n\n{iteratively_binarized_data}\n\n")

    print(Back.BLUE + "Result using vector binarizer:")
    print(f"\n\n{vector_binarized_data}\n\n")


main()
