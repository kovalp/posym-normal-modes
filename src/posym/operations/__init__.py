import numpy as np


def get_cross_distance_table_py(coordinates_1, coordinates_2):

    coordinates_1 = np.array(coordinates_1)
    coordinates_2 = np.array(coordinates_2)

    distances = np.zeros((len(coordinates_1), len(coordinates_2)))

    for i, c1 in enumerate(coordinates_1):
        for j, c2 in enumerate(coordinates_2):
            # print(i, j, c1, c2, np.linalg.norm(c1 - c2))
            distances[i, j] = np.linalg.norm(c1 - c2)
    return distances


def get_permutation_hungarian(distance_table, symbols):
    """
    This function takes distance_table and returns the permutation vector
    that minimizes its trace, using the Hungarian method.
    """
    from scipy.optimize import linear_sum_assignment

    def get_perm_submatrix(sub_matrix):
        row_ind, col_ind = linear_sum_assignment(sub_matrix)
        perm = np.zeros_like(row_ind)
        perm[row_ind] = col_ind
        return perm

    # return get_perm_submatrix(distance_table)

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


class Operation:
    def __init__(self, label):
        self._label = label

    def get_permutation(self, operation, coordinates, symbols, return_dot=False):
        operated_coor = np.dot(operation, coordinates.T).T

        symbols = [int.from_bytes(num.encode(), 'big') for num in symbols]
        distance_table = get_cross_distance_table_py(coordinates, operated_coor)
        perm = get_permutation_hungarian(distance_table, symbols)
        permu_coor = operated_coor[list(perm)]

        if return_dot:
            measure = np.einsum('ij, ij -> ', coordinates, permu_coor)
            # measure = np.trace(np.dot(coordinates, permu_coor.T))
            return measure, list(perm)
        else:
            return list(perm)

    def get_normalization(self, coordinates):

        sum_list = []
        for r1 in coordinates:
            for r2 in coordinates:
                subs = np.subtract(r1, r2)
                sum_list.append(np.dot(subs, subs))
        d = np.average(sum_list)

        return d

    @property
    def label(self):
        return self._label
