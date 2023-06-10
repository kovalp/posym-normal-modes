from posym_normal_modes import SymmetryModes


def test_normal_modes() -> None:
    """."""
    coordinates = [[0.00000, 0.0000000, -0.0808819],
                   [-1.43262, 0.0000000, -1.2823700],
                   [1.43262, 0.0000000, -1.2823700]]

    symbols = ['O', 'H', 'H']

    normal_modes = [[[0., 0., -0.075], [-0.381, -0., 0.593], [0.381, -0., 0.593]],  # mode 1
                    [[-0., -0., 0.044], [-0.613, -0., -0.35], [0.613, 0., -0.35]],  # mode 2
                    [[-0.073, -0., -0.], [0.583, 0., 0.397], [0.583, 0., -0.397]]]  # mode 3

    sym_modes_gs = SymmetryModes(group='c2v', coordinates=coordinates, modes=normal_modes,
                                 symbols=symbols)

    for i, mode_ref in enumerate(('A1', 'A1', 'B2')):
        assert str(sym_modes_gs.get_state_mode(i)) == mode_ref
