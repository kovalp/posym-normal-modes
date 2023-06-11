import numpy as np
import pytest


def test_einsum() -> None:

    coo = np.array([[1.40000, 2.0000000, -3.0808819],
                    [-1.43262, 1.2000, -1.2823700],
                    [1.43262, 0.120000, -1.2823700]])

    doo = np.array([[1.900, 1.1000, -3.819],
                    [-1.43, 1.6000, -1.370],
                    [1.362, 0.2100, -1.700]])

    res = np.einsum('ij, ij -> ', coo, doo)
    np.multiply(coo, doo, out=coo)
    assert coo.sum() == pytest.approx(res)

