"""Scaling matrices with Python.

This file contains a number of basic examples of how scaling can be 
implemented using numpy and sklearn packages.

Typical usage example:
    result = minmax_scale(matrix, 0, 1)
    diy_result = diy_minmax_scale(matrix, 0, 1)
"""

import colorama
import numpy as np
from colorama import Back
from pyfiglet import Figlet
from sklearn import preprocessing


def minmax_scale(matrix: np.array, lower: float, upper: float) -> np.array:
    """Min-Max scale with sklearn

    Transform values in a matrix such that the maximum value in a row is assigned
    a value of 1, the lowest is assigned 0 and all other columns in that row are
    relative to that.

    Args:
        matrix (np.array): A numpy array of input values.

    Returns:
        An output array with the transformed elements.
    """
    # Creates the transformation function
    data_scaler_minmax = preprocessing.MinMaxScaler(
        feature_range=(lower, upper)
    )
    # Applies the transformation function to the matrix
    result = data_scaler_minmax.fit_transform(matrix)
    return result


def diy_minmax_scale(matrix: np.array, lower: float, upper: float) -> np.array:
    """DIY implementation of min-max scale

    Transform values in a matrix such that the maximum value in a row is assigned
    a value of 1, the lowest is assigned 0 and all other columns in that row are
    relative to that.

    Args:
        matrix (np.array): A numpy array of input values.
        lower (float): The desired lowest value
        upper (float): The desired highest value

    Returns:
        An output array with the transformed elements.
    """
    result = np.empty(matrix.shape)
    col_maxes = np.amax(matrix, axis=0)
    col_mins = np.amin(matrix, axis=0)
    for row_index, row in enumerate(matrix):
        for col_index, _ in enumerate(row):
            col_max = col_maxes[col_index]
            col_min = col_mins[col_index]
            unit_value = (matrix[row_index][col_index] - col_min) / (
                col_max - col_min
            )
            result[row_index][col_index] = (upper - lower) * unit_value + lower
    return result


def main() -> None:
    """Performs a demonstration of the above functions"""
    colorama.init(autoreset=True)

    figlet = Figlet()
    print(figlet.renderText("Min-Max Scaling"))

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

    result: np.array = minmax_scale(matrix, 0, 1)
    diy_result: np.array = diy_minmax_scale(matrix, 0, 1)

    print(Back.GREEN + "Result using sklearn:")
    print(f"\n\n{result}\n\n")

    print(Back.MAGENTA + "Result using DIY iteration:")
    print(f"\n\n{diy_result}\n\n")


main()
