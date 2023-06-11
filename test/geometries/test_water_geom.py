import numpy as np

from posym_normal_modes.symmetry.molecule_base import SymmetryMoleculeBase


def test_water() -> None:
    """."""
    symbols = ['O', 'H', 'H']
    coordinates = [[0.00000, 0.0000000, -0.0808819],
                   [-1.43262, 0.0000000, -1.2823700],
                   [1.43262, 0.0000000, -1.2823700]]

    coo = np.array(coordinates)
    coo = coo - coo.mean(axis=0)
    print()
    print(coo.mean(axis=0))

    sym_geom = SymmetryMoleculeBase(group='Td', coordinates=coo, symbols=symbols)
    print('Symmetry measure Td : ', sym_geom.measure)

    sym_geom = SymmetryMoleculeBase(group='C2v', coordinates=coo, symbols=symbols)
    print('Symmetry measure C2v : ', sym_geom.measure)

    sym_geom = SymmetryMoleculeBase(group='C3v', coordinates=coo, symbols=symbols)
    print('Symmetry measure C3v : ', sym_geom.measure)

    sym_geom = SymmetryMoleculeBase(group='C4v', coordinates=coo, symbols=symbols)
    print('Symmetry measure C4v : ', sym_geom.measure)

