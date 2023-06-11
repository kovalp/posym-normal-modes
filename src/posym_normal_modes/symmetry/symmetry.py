__author__ = 'Abel Carreras'
__version__ = '0.5.4'

from scipy.spatial.transform import Rotation
import numpy as np
import pandas as pd

from posym_normal_modes.symmetry.molecule_base import SymmetryMoleculeBase


class SymmetryFunction(SymmetryMoleculeBase):
    def __init__(self, group, function, orientation_angles=None, center=None):
        """Constructor.

        Args:
            group:
            function:
            orientation_angles:
            center:
        """
        symbols, coordinates = function.get_environment_centers()
        self._setup_structure(coordinates, symbols, group, center, orientation_angles)
        self._function = function.copy()
        self._function.apply_translation(-np.array(self._center))
        rot = Rotation.from_euler('zyx', self._angles, degrees=True)
        self._self_similarity = (self._function * self._function).integrate
        self._operator_measures = []
        for operation in self._pg.operations:
            operator_measures = []
            for op in self._pg.get_sub_operations(operation.label):
                overlap = op.get_overlap_func(self._function, self._function, orientation=rot)
                operator_measures.append(overlap/self._self_similarity)

            self._operator_measures.append(np.array(operator_measures))

        total_state = pd.Series(self._operator_measures, index=self._pg.op_labels)
        zero3 = np.zeros(3)
        super().__init__(group, self._coordinates, self._symbols, total_state, self._angles, zero3)

    @property
    def self_similarity(self):
        return self._self_similarity


