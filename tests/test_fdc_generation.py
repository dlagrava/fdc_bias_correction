import pytest
from context import statistical_transformations
import numpy as np


def test_fdc_function():
    from statistical_transformations.fdc_utils import calculate_FDC

    flow_values = np.array([1.0] * 100)
    probabilities_fdc = np.array([0.1, 0.5, 0.9])
    expected_values = np.array([1.0] * 3)
    fdc = calculate_FDC(probabilities_fdc, flow_values)
    assert np.isclose(expected_values, fdc), "Constant test not OK"

    flow_values = np.array([1.0] * 50 + [0.0] * 50)
    probabilities_fdc = np.array([0.1, 0.5, 0.9])
    expected_values = np.array([1.0, 0.5, 0.0])
    fdc = calculate_FDC(probabilities_fdc, flow_values)
    assert np.isclose(expected_values, fdc), "Binary test not OK"
