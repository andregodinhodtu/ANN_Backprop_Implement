import sys
sys.path.append("src/numpy_version")

import pytest
import numpy as np
from ANN_numpy import ANN_numpy as ANN


# ============================================================
# Helper: assert both shape and values
# ============================================================

def assert_equal(actual, expected):
    """Assert that actual and expected have the same shape and close values."""
    actual_arr = np.array(actual)
    expected_arr = np.array(expected)
    assert actual_arr.shape == expected_arr.shape, (
        f"Shape mismatch: got {actual_arr.shape}, expected {expected_arr.shape}"
    )
    assert np.allclose(actual_arr, expected_arr)


# ============================================================
# _build_ANN — happy path
# ============================================================

#def test_build_standart_ANN():
    


# ============================================================
# weights_matrix setter — TypeError
# ============================================================


# ============================================================
# weights_matrix setter — ValueError
# ============================================================
