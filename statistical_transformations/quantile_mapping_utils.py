import xarray as xr
import numpy as np
from utils import select_valid_years

attrs_station_rchid = {"long_name": "REC identifier for the stream reach on which station is located",
                       "valid_min": 0, "valid_max": 2000000000}
attrs_rchid = {'long_name': 'identifier of selected reaches (REC)'}
attrs_quantile_sim = {'description': 'Quantiles for simulation', 'standard_name': 'Q_Sim',
                      'long_name': 'Quantile mapping using hourly data', 'units': 'meter3 second-1', 'valid_min': 0.0,
                      'valid_max': 1e+20}
attrs_quantile_obs = {'description': 'Quantiles for observations', 'standard_name': 'Q_Obs',
                      'long_name': 'Quantile mapping using hourly data', 'units': 'meter3 second-1', 'valid_min': 0.0,
                      'valid_max': 1e+20}
attrs_quantiles = {'standard_name': 'Quantile', 'long_name': 'Quantiles for sampling', 'valid_min': 0.0,
                   'valid_max': 1.0, 'units': '[0..1]'}


def create_quantile_dict_sim(DS: xr.Dataset, number_of_quantiles=1001, output_variable_name: str = "Sim_Quantile"):
    """

    Parameters
    ----------
    DS
    number_of_quantiles
    output_variable_name

    Returns
    -------

    """
    reaches = DS.rchid.values
    flow_variable_name = "mod_streamq"
    start_prob = 0.0
    end_prob = 1.0

    quantiles = np.linspace(start_prob, end_prob, number_of_quantiles)

    print("Calculation of simulated Quantiles")
    data = []
    for idx, reach_id in enumerate(reaches):
        print(reach_id)
        dimensions_flow_variable = len(DS.variables[flow_variable_name].dims)

        flow_values = np.ndarray([])
        if dimensions_flow_variable == 2:
            flow_values = DS.variables[flow_variable_name][:, idx].values
        elif dimensions_flow_variable == 4:
            flow_values = DS.variables[flow_variable_name][:, idx, 0, 0].values
        else:
            print("Flow values variable has a weird number of dimensions: ", dimensions_flow_variable)
            exit(1)

        values_quantiles = np.quantile(flow_values, quantiles)

        data.append(values_quantiles)

    quantile_dict = {
        'rchid': {'dims': ('nrch',), 'data': reaches, 'attrs': attrs_rchid},
        output_variable_name: {'dims': ('nrch', 'quantile'), 'data': data, 'attrs': attrs_quantile_sim},
        'quantile': {'dims': ('quantile',), 'data': quantiles},
    }

    return quantile_dict


def create_quantile_dict_obs(DS: xr.Dataset, number_of_quantiles=1001,
                             output_variable_name: str = "Obs_Quantile"):
    """


    Parameters
    ----------
    DS
    number_of_quantiles
    output_variable_name

    Returns
    -------

    """
    station_rchids = DS.station_rchid.values.astype(int)

    start_prob = 0.0
    end_prob = 1.0

    quantiles = np.linspace(start_prob, end_prob, number_of_quantiles)

    data = []
    print("Calculation of observed FDC")

    good_stations = []
    for station_idx, station_rchid in enumerate(station_rchids):
        print(station_rchid)

        flow_values = select_valid_years(DS, station_idx)
        if len(flow_values) > 0:
            values_quantiles = np.quantile(flow_values, quantiles)
            data.append(values_quantiles)
            good_stations.append(station_rchid)
        else:
            print("Ignoring...")

    quantile_dict = {
        'station_rchid': {'dims': ('station',), 'data': good_stations, "attrs": attrs_station_rchid},
        output_variable_name: {'dims': ('station', 'quantile'), 'data': data, 'attrs': attrs_quantile_obs},
        'percentile': {'dims': ('quantile',), 'data': quantiles, 'attrs': attrs_quantiles},
    }

    return quantile_dict
