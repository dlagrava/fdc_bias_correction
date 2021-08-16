import pytest
from context import statistical_transformations
import numpy as np
import xarray as xr


def create_fake_DS():
    import scipy.stats
    normal_values = scipy.stats.norm.rvs(size=(1000000,1))
    data = normal_values
    d = {
        "mod_streamq": {"dims": ("t", "nrch",), "data": data},
        "rchid": {"dims": ("nrch",), "data": [100000]},
    }

    ds = xr.Dataset.from_dict(d)

    return ds


def test_quantile_generation():
    from statistical_transformations.quantile_mapping_utils import create_quantile_dict_sim


def test_quantile_bias_correction():
    pass
