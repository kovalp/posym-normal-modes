from typing import List, Tuple

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def get_perm_sub_matrix(sub_matrix: np.ndarray):
    row_ind, col_ind = linear_sum_assignment(sub_matrix)
    perm = np.zeros_like(row_ind)
    perm[row_ind] = col_ind
    return perm


def get_permutation_given_sub_matrices(distance_table: np.ndarray,
                                       sub_matrices_indices: List[List[int]]
                                       ) -> np.ndarray:
    # Determine the permutation for each sub matrix.
    perm_sub_matrices = []
    for index in sub_matrices_indices:
        sub_matrix = distance_table[index, :][:, index]
        perm_sub = get_perm_sub_matrix(sub_matrix)
        perm_sub_matrices.append(perm_sub)

    # Restore global permutation by joining permutations of sub matrices.
    global_permutation = np.zeros(len(distance_table), dtype=int)
    for index, perm in zip(sub_matrices_indices, perm_sub_matrices):
        global_permutation[index] = index[perm]

    return global_permutation


def get_sub_matrices_for_hungarian_permutation(int_symbols: List[int]) -> List[np.ndarray]:
    """Generate auxiliary matrices for the Hungarian permutation method.

    Args:
        int_symbols: chemical symbols as integers.

    Returns:
        The list of integers.
    """
    sub_matrices_indices = []
    for s in np.unique(int_symbols):
        sub_matrices_indices.append(np.array([j for j, s2 in enumerate(int_symbols) if s2 == s]))
    return sub_matrices_indices


def get_permutation_hungarian(distance_table: np.ndarray, symbols: List[int]) -> np.ndarray:
    """Compose a permutation vector that minimises the trace of distance table.

    This function takes distance_table and returns the permutation vector
    that minimises its trace, using the Hungarian method.

    Args:
        distance_table:
        symbols:

    Returns:
        The permutation vector.
    """
    sub_matrices_indices = get_sub_matrices_for_hungarian_permutation(symbols)
    return get_permutation_given_sub_matrices(distance_table, sub_matrices_indices)


def get_permutation_and_transformed_coordinates(operation: np.ndarray,
                                                coordinates: np.ndarray,
                                                symbols: List[str]
                                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Compose a permutation of ?

    Args:
        operation: the matrix representing a symmetry operation.
        coordinates: the (Cartesian) coordinates of the atoms. The shape is (num_atoms, 3).
        symbols: the list of chemical symbols ... this will change...

    Returns:
        Permutation and transformed coordinates.
    """
    operated_coo = np.dot(operation, coordinates.T).T
    symbols = [int.from_bytes(num.encode(), 'big') for num in symbols]
    distance_table = cdist(coordinates, operated_coo)
    permutation = get_permutation_hungarian(distance_table, symbols)
    return permutation, operated_coo


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
    perm, _coo = get_permutation_and_transformed_coordinates(operation, coordinates, symbols)
    return perm


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
    perm, coo = get_permutation_and_transformed_coordinates(operation, coordinates, symbols)
    permuted_coordinates = coo[perm]
    np.multiply(coordinates, permuted_coordinates, out=permuted_coordinates)
    return permuted_coordinates.sum()


def get_int_symbols(symbols: List[str]) -> List[int]:
    """Convert the chemical symbols to integers.

    Args:
        symbols: the list of chemical symbols, e.g. ['H', 'H', 'O'].

    Returns:
        The list of integers.
    """
    return [int.from_bytes(num.encode(), 'big') for num in symbols]


class Permutations:
    """This class would cache repetitive operations while composing optimal permutations."""
    def __init__(self, coordinates: np.ndarray, symbols: List[str]) -> None:
        """Constructor.

        Args:
            coordinates: the coordinates of atoms to be permuted.
            symbols: the chemical symbols of atoms.
        """
        self.coordinates = coordinates
        self.coordinates_transposed = np.transpose(coordinates)
        self.symbols = symbols
        self.int_symbols = get_int_symbols(symbols)
        self.sub_matrices = get_sub_matrices_for_hungarian_permutation(self.int_symbols)
        self.norm: float = np.einsum('ij, ij -> ', self.coordinates, self.coordinates)

    def get_perm_and_trans_coo(self, operation: np.ndarray) -> Tuple[float, np.ndarray]:
        """

        Args:
            operation:

        Returns:

        """
        operated_coo = np.dot(operation, self.coordinates_transposed).T
        distance_table = cdist(self.coordinates, operated_coo)
        permutation = get_permutation_given_sub_matrices(distance_table, self.sub_matrices)
        return permutation, operated_coo

    def get_measure(self, operation: np.ndarray) -> float:
        """

        Args:
            operation:

        Returns:

        """
        perm, coo = self.get_perm_and_trans_coo(operation)
        permuted_coordinates = coo[perm]
        np.multiply(self.coordinates, permuted_coordinates, out=permuted_coordinates)
        return permuted_coordinates.sum()

    def get_permutation(self, operation: np.ndarray) -> np.ndarray:
        """

        Args:
            operation:

        Returns:

        """
        perm, _coo = self.get_perm_and_trans_coo(operation)
        return perm
