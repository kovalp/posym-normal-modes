import copy
import itertools

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from posym_normal_modes.pointgroup import PointGroup
from posym_normal_modes.symmetry.base import SymmetryBase
from posym_normal_modes.permutations import Permutations

cache_orientation = {}


def get_hash(coordinates, symbols, group):
    return hash((np.array2string(coordinates, precision=2), tuple(symbols), group))


class SymmetryMoleculeBase(SymmetryBase):
    def __init__(self, group, coordinates, symbols, total_state=None, orientation_angles=None,
                 center=None, fast_optimization=True):
        """Constructor.

        Args:
            group:
            coordinates:
            symbols:
            total_state:
            orientation_angles:
            center:
            fast_optimization:
        """
        self.permutations = None
        self._setup_structure(coordinates, symbols, group, center, orientation_angles,
                              fast_optimization=fast_optimization)

        if total_state is None:
            rot = Rotation.from_euler('zyx', self._angles, degrees=True)
            self._operator_measures = []
            for operation in self._pg.operations:
                operator_measures = []
                for op in self._pg.get_sub_operations(operation.label):
                    overlap = op.get_measure_pos(self.permutations, orientation=rot)
                    operator_measures.append(overlap)

                self._operator_measures.append(np.array(operator_measures))

            total_state = pd.Series(self._operator_measures, index=self._pg.op_labels)

        super().__init__(group, total_state)

    def _setup_structure(self, coordinates, symbols, group, center, orientation_angles,
                         fast_optimization=True):

        self._coordinates = np.array(coordinates)
        self.permutations = Permutations(self._coordinates, symbols)
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

    def optimization_function_simple(self, angles):
        """
        This function uses only one operation of each type (described in the IR table).
        This approach works well when the molecule has a symmetry close to the group
        """
        rot = Rotation.from_euler('zyx', angles, degrees=True)

        measures = []
        for operation in self._pg.operations:
            measure = operation.get_measure_pos(self.permutations,
                                                orientation=rot, normalized=False)
            measures.append(measure)

        # get most symmetric IR value
        return -np.dot(measures, self._pg.trans_matrix_inv[0])

    def optimization_function_full(self, angles):
        """
        This function uses all operations of the group and averages the overlap of equivalent
        operations
        """

        rot = Rotation.from_euler('zyx', angles, degrees=True)

        operator_measures = []
        for operation in self._pg.operations:
            sub_operator_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                overlap = op.get_measure_pos(self.permutations, orientation=rot)
                sub_operator_measures.append(overlap)
            operator_measures.append(np.average(sub_operator_measures))

        # get most symmetric IR value
        return -np.dot(operator_measures, self._pg.trans_matrix_inv[0])

    def get_orientation(self, fast_optimization=True, scan_step=30):
        """
        get orientation angles for optimum orientation.
        Use full=False to orient perfect symmetric molecules.
        Use full=True to orient quasi symmetric molecules

        :param fast_optimization: if True use only a subset of symmetry elements
                                (for exact symmetry objets)
        :param scan_step: step angle (deg) use for the preliminary scan
        :return:
        """
        hash_num = get_hash(self._coordinates, self._symbols, self._pg.label)
        if hash_num in cache_orientation:
            return cache_orientation[hash_num]

        # Define whether to use simple function (faster) or full (slower)
        optimization_function = (self.optimization_function_simple if fast_optimization else
                                 self.optimization_function_full)

        # Preliminary scan
        ranges = np.arange(-90, 90+scan_step, scan_step)
        guess_angles = ref_value = None
        for angles in itertools.product(ranges, ranges, ranges):
            value = optimization_function(angles)
            if ref_value is None or value < ref_value:
                ref_value = value
                guess_angles = angles

        result = minimize(optimization_function, guess_angles, method='CG')

        cache_orientation[hash_num] = result.x
        return cache_orientation[hash_num]

    def get_oriented_operations(self):
        rotmol = Rotation.from_euler('zyx', self.orientation_angles, degrees=True)

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
        rot = Rotation.from_euler('zyx', self._angles, degrees=True)

        operator_measures = []
        for operation in self._pg.operations:
            sub_operator_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                overlap = op.get_measure_pos(self.permutations, orientation=rot)
                sub_operator_measures.append(overlap)
            operator_measures.append(np.average(sub_operator_measures))

        # get most symmetric IR value
        ir_rep_diff = np.dot(operator_measures, self._pg.trans_matrix_inv[0])

        return 100 * (1 - ir_rep_diff)

    @property
    def opt_coordinates(self):
        rot = Rotation.from_euler('zyx', self._angles, degrees=True)
        return rot.apply(self._coordinates)

    @property
    def orientation_angles(self):
        return self._angles

    @property
    def center(self):
        return self._center

