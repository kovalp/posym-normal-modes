__author__ = 'Abel Carreras'
__version__ = '0.5.4'

from posym_normal_modes.tools import list_round
from posym_normal_modes.pointgroup import PointGroup
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize
import numpy as np
import pandas as pd
import itertools


cache_orientation = {}
def get_hash(coordinates, symbols, group):
    return hash((np.array2string(coordinates, precision=2), tuple(symbols), group))


class SymmetryBase():
    """
    This class is supposed to be used as a base for more complex symmetry objects

    """
    def __init__(self, group, rep, normalize=False):

        self._pg = PointGroup(group)
        self._group = group.lower()

        if isinstance(rep, str):
            if rep not in self._pg.ir_labels:
                raise Exception('Representation do not match with group\nAvailable: {}'.format(self._pg.ir_labels))
            self._op_representation = self._pg.ir_table[rep]
            #if normalize:
            #    self._op_representation /= self._pg.ir_table[rep]['E']

        elif isinstance(rep, pd.Series):
            if np.all(self._pg.ir_table.sort_index().index == rep.sort_index().index):
                self._op_representation = rep.reindex(self._pg.ir_table.index)
            else:
                raise Exception('Representation not in group')

        if normalize:
            op_rep = np.dot(self._pg.trans_matrix_norm, np.dot(self._pg.trans_matrix_inv, self._op_representation.values))
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


class SymmetryMoleculeBase(SymmetryBase):
    def __init__(self, group, coordinates, symbols, total_state=None, orientation_angles=None, center=None,
                 fast_optimization=True):

        self._setup_structure(coordinates, symbols, group, center, orientation_angles, fast_optimization=fast_optimization)

        if total_state is None:
            rotmol = R.from_euler('zyx', self._angles, degrees=True)
            self._operator_measures = []
            for operation in self._pg.operations:
                operator_measures = []
                for op in self._pg.get_sub_operations(operation.label):
                    overlap = op.get_measure_pos(self._coordinates, symbols, orientation=rotmol)
                    operator_measures.append(overlap)

                self._operator_measures.append(np.array(operator_measures))

            total_state = pd.Series(self._operator_measures, index=self._pg.op_labels)

        super().__init__(group, total_state)

    def _setup_structure(self, coordinates, symbols, group, center, orientation_angles, fast_optimization=True):

        self._coordinates = np.array(coordinates)
        self._symbols = symbols
        self._pg = PointGroup(group)

        if '_center' not in self.__dir__():
            self._center = center
            if self._center is None:
                self._center = np.average(self._coordinates, axis=0)

            self._coordinates = np.array([c - self._center for c in self._coordinates])

        if orientation_angles is None:
            self._angles = self.get_orientation(fast_optimization=fast_optimization)
        else:
            self._angles = orientation_angles

    def get_orientation(self, fast_optimization=True, scan_step=10):
        """
        get orientation angles for optimum orientation.
        Use full=False to orient perfect symmetric molecules. Use full=True to orient quasi symmetric molecules

        :param fast_optimization: if True use only a subset of symmetry elements (for exact symmetry objets)
        :param scan_step: step angle (deg) use for the preliminary scan
        :return:
        """

        hash_num = get_hash(self._coordinates, self._symbols, self._pg.label)
        if hash_num in cache_orientation:
            return cache_orientation[hash_num]

        def optimization_function_simple(angles):
            """
            This function uses only one operation of each type (described in the IR table).
            This approach works well when the molecule has a symmetry close to the group
            """

            rotmol = R.from_euler('zyx', angles, degrees=True)

            coor_measures = []
            for operation in self._pg.operations:
                coor_m = operation.get_measure_pos(self._coordinates, self._symbols, orientation=rotmol, normalized=False)
                coor_measures.append(coor_m)

            # get most symmetric IR value
            return -np.dot(coor_measures, self._pg.trans_matrix_inv[0])

        def optimization_function_full(angles):
            """
            This function uses all operations of the group and averages the overlap of equivalent operations
            """

            rotmol = R.from_euler('zyx', angles, degrees=True)

            operator_measures = []
            for operation in self._pg.operations:
                sub_operator_measures = []
                for op in self._pg.get_sub_operations(operation.label):
                    overlap = op.get_measure_pos(self._coordinates, self._symbols, orientation=rotmol)
                    sub_operator_measures.append(overlap)
                operator_measures.append(np.average(sub_operator_measures))

            # get most symmetric IR value
            return -np.dot(operator_measures, self._pg.trans_matrix_inv[0])

        # define if use simple function (faster) or full (slower)
        optimization_function = optimization_function_simple if fast_optimization else optimization_function_full

        # preliminary scan
        ranges = np.arange(-90, 90+scan_step, scan_step)
        guess_angles = ref_value = None
        for angles in itertools.product(ranges, ranges, ranges):
            value = optimization_function(angles)
            if ref_value is None or value < ref_value:
                ref_value = value
                guess_angles = angles

        result = minimize(optimization_function, guess_angles, method='CG',)

        cache_orientation[hash_num] = result.x
        return cache_orientation[hash_num]

    def get_oriented_operations(self):
        import copy
        rotmol = R.from_euler('zyx', self.orientation_angles, degrees=True)

        operations_list = []
        for operation in self._pg.operations:
            for sub_operation in self._pg.get_sub_operations(operation.label):
                sub_operation = copy.deepcopy(sub_operation)
                sub_operation.apply_rotation(rotmol)
                operations_list.append(sub_operation)

        return operations_list

    @property
    def measure(self):
        return 100*(1-self.get_ir_representation().values[0])

    @property
    def measure_pos(self):

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        operator_measures = []
        for operation in self._pg.operations:
            sub_operator_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                overlap = op.get_measure_pos(self._coordinates, self._symbols, orientation=rotmol)
                sub_operator_measures.append(overlap)
            operator_measures.append(np.average(sub_operator_measures))

        # get most symmetric IR value
        ir_rep_diff = np.dot(operator_measures, self._pg.trans_matrix_inv[0])

        return 100 * (1 - ir_rep_diff)

    @property
    def opt_coordinates(self):
        rotmol = R.from_euler('zyx', self._angles, degrees=True)
        return rotmol.apply(self._coordinates)

    @property
    def orientation_angles(self):
        return self._angles

    @property
    def center(self):
        return self._center


class SymmetryModes(SymmetryMoleculeBase):
    def __init__(self, group, coordinates, modes, symbols, orientation_angles=None, center=None):

        self._setup_structure(coordinates, symbols, group, center, orientation_angles)

        self._modes = modes

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        self._mode_measures = []
        for operation in self._pg.operations:
            mode_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                mode_m = op.get_measure_modes(self._coordinates, self._modes, self._symbols, orientation=rotmol)
                mode_measures.append(mode_m)

            mode_measures = np.array(mode_measures)
            self._mode_measures.append(mode_measures)

        mode_measures_total = []
        for op in self._mode_measures:
            op_list = []
            for m in op:
                op_list.append(sum(m))
            mode_measures_total.append(op_list)

        # reshape mode measures
        reshaped_modes_measures = []
        for m in range(len(self._mode_measures[0].T)):
            reshaped_modes_measures.append([k[:, m] for k in self._mode_measures])

        self._mode_measures = reshaped_modes_measures

        total_state = pd.Series(mode_measures_total, index=self._pg.op_labels)

        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0, 0, 0])

    def get_state_mode(self, n):
        return SymmetryBase(group=self._group, rep=pd.Series(self._mode_measures[n],
                                                             index=self._pg.op_labels))


class SymmetryModesFull(SymmetryMoleculeBase):
    def __init__(self, group, coordinates, symbols, orientation_angles=None):

        self._setup_structure(coordinates, symbols, group, None, orientation_angles)

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        trans_rots = []
        for label in self._pg.ir_table.rotations + self._pg.ir_table.translations:
            trans_rots.append(self._pg.ir_table[label].values)

        trans_rots = np.sum(trans_rots, axis=0)

        self._mode_measures = []
        for operation in self._pg.operations:
            mode_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                measure_xyz = op.get_measure_xyz(orientation=rotmol)
                measure_atom = op.get_measure_atom(self._coordinates, self._symbols, orientation=rotmol)

                mode_measures.append(measure_xyz * measure_atom)

            mode_measures = np.array(mode_measures)
            self._mode_measures.append(mode_measures)

        self._mode_measures = np.array(self._mode_measures, dtype=object).flatten() - trans_rots
        total_state = pd.Series(self._mode_measures, index=self._pg.op_labels)
        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0, 0, 0])


class SymmetryFunction(SymmetryMoleculeBase):
    def __init__(self, group, function, orientation_angles=None, center=None):

        symbols, coordinates = function.get_environment_centers()

        self._setup_structure(coordinates, symbols, group, center, orientation_angles)

        self._function = function.copy()
        self._function.apply_translation(-np.array(self._center))

        rotmol = R.from_euler('zyx', self._angles, degrees=True)

        self._self_similarity = (self._function * self._function).integrate

        self._operator_measures = []
        for operation in self._pg.operations:
            operator_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                overlap = op.get_overlap_func(self._function, self._function, orientation=rotmol)
                operator_measures.append(overlap/self._self_similarity)

            self._operator_measures.append(np.array(operator_measures))

        total_state = pd.Series(self._operator_measures, index=self._pg.op_labels)
        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, [0, 0, 0])

    @property
    def self_similarity(self):
        return self._self_similarity


