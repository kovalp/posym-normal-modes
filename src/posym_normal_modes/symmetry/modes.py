import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from posym_normal_modes.symmetry.base import SymmetryBase
from posym_normal_modes.symmetry.molecule_base import SymmetryMoleculeBase


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
