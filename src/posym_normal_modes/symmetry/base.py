import numpy as np
import pandas as pd

from posym_normal_modes.pointgroup import PointGroup
from posym_normal_modes.tools import list_round


class SymmetryBase:
    """This class is supposed to be used as a base for more complex symmetry objects."""
    def __init__(self, group, rep, normalize=False):

        self._pg = PointGroup(group)
        self._group = group.lower()

        if isinstance(rep, str):
            if rep not in self._pg.ir_labels:
                raise ValueError('Representation do not match with group\n'
                                 f'Available: {self._pg.ir_labels}')
            self._op_representation = self._pg.ir_table[rep]

        elif isinstance(rep, pd.Series):
            if np.all(self._pg.ir_table.sort_index().index == rep.sort_index().index):
                self._op_representation = rep.reindex(self._pg.ir_table.index)
            else:
                raise ValueError('Representation is not in group.')

        if normalize:
            op_rep = np.dot(self._pg.trans_matrix_norm,
                            np.dot(self._pg.trans_matrix_inv, self._op_representation.values))
            self._op_representation = pd.Series(op_rep, index=self._pg.op_labels)

    def get_reduced_op_representation(self):
        red_values = []
        for value in self._op_representation.values:
            red_values.append(np.average(value))
        return pd.Series(red_values, index=self._op_representation.index)

    def get_op_representation(self):
        return self._op_representation

    def get_ir_representation(self):
        ir_rep = np.dot(self._pg.trans_matrix_inv, self.get_reduced_op_representation().values)
        return pd.Series(ir_rep, index=self._pg.ir_labels)

    def get_point_group(self):
        return self._pg

    def __str__(self):

        ir_rep = self.get_ir_representation().values
        ir_rep = list_round(ir_rep)

        ir_labels = self.get_ir_representation().index

        str = ''
        for i, r in enumerate(ir_rep):
            if np.add.reduce(np.square(ir_rep[:i])) > 0 and r > 0:
                    str += ' + '
            elif r < 0:
                    str += ' - '
            if np.abs(r - 1) < 2e-2:
                str += ir_labels[i]
            elif np.abs(r) > 0:
                str += '{} {}'.format(abs(r), ir_labels[i])

        return str

    def __repr__(self):
        ir_rep = self.get_ir_representation().values
        ir_rep = list_round(ir_rep)

        ir_labels = self.get_ir_representation().index

        txt = ''
        for i, r in enumerate(ir_rep):
            # print('>> ', np.add.reduce(ir_rep[:i]**2), 'r:', r)
            if np.add.reduce(np.square(ir_rep[:i])) > 0 and r > 0 and len(ir_rep[:i]) > 0:
                txt += '+'
            if r == 1:
                txt += ir_labels[i]
            elif r > 0:
                txt += '{}{}'.format(r, ir_labels[i])
        return txt

    def __add__(self, other):

        if self._group == other._group:
            return SymmetryBase(self._group,
                                self._op_representation + other._op_representation)

        raise Exception('Incompatible point groups')

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, other):

        if isinstance(other, (float, int)):
            return SymmetryBase(self._group,
                                self._op_representation * other)

        elif isinstance(other, SymmetryBase):
            mul_rep = self._op_representation * other._op_representation

            return SymmetryBase(self._group, mul_rep)
        else:
            raise Exception('Symmetry operation not possible')
