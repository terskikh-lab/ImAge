# Import libraries
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from numpy.typing import ArrayLike


def distance_matrix_pdist(data: ArrayLike, metric: str = "euclidean"):
    """Takes a dataframe of z-scores and calculates the euclidean distance matrix.
    Parameters: df_zscore = dataframe of zscores
    Returns: df_distance = dataframe of distance matrix in squareform"""
    # create the distance matrix by using the sklearn pdist function in square form
    # this function computes the euclidean distances in an efficient way (1/2 the matrix) and then squareform() reflects it to get the final matrix
    # set the column labels and index to the df_zscore index, ie the trials (see above)
    squareform_data = squareform(pdist(data, metric=metric))
    return squareform_data
