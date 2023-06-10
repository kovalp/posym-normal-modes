from typing import List

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def get_perm_submatrix(sub_matrix):
    row_ind, col_ind = linear_sum_assignment(sub_matrix)
    perm = np.zeros_like(row_ind)
    perm[row_ind] = col_ind
    return perm


def get_permutation_hungarian(distance_table, symbols):
    """
    This function takes distance_table and returns the permutation vector
    that minimizes its trace, using the Hungarian method.
    """
    # separate distance_table in submatrices corresponding to a single symbol
    submatrices_indices = []
    for s in np.unique(symbols):
        submatrices_indices.append([j for j, s2 in enumerate(symbols) if s2 == s])

    # determine the permutation for each submatrix
    perm_submatrices = []
    for index in submatrices_indices:
        submatrix = np.array(distance_table)[index, :][:, index]
        perm_sub = get_perm_submatrix(submatrix)
        perm_submatrices.append(perm_sub)

    # restore global permutation by joining permutations of submatrices
    global_permutation = np.zeros(len(distance_table), dtype=int)
    for index, perm in zip(submatrices_indices, perm_submatrices):
        index = np.array(index)
        global_permutation[index] = index[perm]

    return global_permutation


def get_permutation(operation: np.ndarray,
                    coordinates: np.ndarray,
                    symbols: List[str]) -> np.ndarray:
    """Compose a permutation of ?

    Args:
        operation:
        coordinates:
        symbols:

    Returns:

    """
    operated_coo = np.dot(operation, coordinates.T).T
    symbols = [int.from_bytes(num.encode(), 'big') for num in symbols]
    distance_table = cdist(coordinates, operated_coo)
    return get_permutation_hungarian(distance_table, symbols)


def get_measure(operation: np.ndarray,
                coordinates: np.ndarray,
                symbols: List[str]):
    """Compose a permutation of ?

    Args:
        operation:
        coordinates:
        symbols:

    Returns:

    """
    operated_coo = np.dot(operation, coordinates.T).T
    symbols = [int.from_bytes(num.encode(), 'big') for num in symbols]
    distance_table = cdist(coordinates, operated_coo)
    permutation = get_permutation_hungarian(distance_table, symbols)
    permuted_coordinates = operated_coo[permutation]
    return np.einsum('ij, ij -> ', coordinates, permuted_coordinates)
