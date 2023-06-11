import numpy as np

from posym_normal_modes.symmetry.molecule_base import SymmetryMoleculeBase

coordinates = [[0.0000000000, 0.0000000000, 0.0000000000],
               [0.5541000000, 0.7996000000, 0.4965000000],
               [0.6833000000, -0.8134000000, -0.2536000000],
               [-0.7782000000, -0.3735000000, 0.6692000000],
               [-0.4593000000, 0.3874000000, -0.9121000000]]

positions = [[-1.2700e-02, 1.0858e+00, 8.0000e-03],
             [2.1000e-03, -4.1000e-03, 2.0000e-03],
             [1.0099e+00, 1.4631e+00, 3.0000e-04],
             [-5.3990e-01, 1.4469e+00, -8.7510e-01],
             [-5.2290e-01, 1.4373e+00, 9.0480e-01]]

symbols = ['C', 'H', 'H', 'H', 'H']


def test_methane() -> None:
    """."""
    coo = np.array(positions)

    print()
    print(coo.sum(axis=0))

    sym_geom = SymmetryMoleculeBase(group='Td', coordinates=coo, symbols=symbols)
    print('Symmetry measure Td : ', sym_geom.measure)

    sym_geom = SymmetryMoleculeBase(group='C3v', coordinates=coo, symbols=symbols)
    print('Symmetry measure C3v : ', sym_geom.measure)

    sym_geom = SymmetryMoleculeBase(group='C4v', coordinates=coo, symbols=symbols)
    print('Symmetry measure C4v : ', sym_geom.measure)
