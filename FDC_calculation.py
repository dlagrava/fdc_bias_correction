import xarray as xr
import numpy as np
from utils import select_valid_years

attrs_station_rchid = {"long_name": "REC identifier for the stream reach on which station is located",
                       "valid_min": 0, "valid_max": 2000000000}
attrs_rchid = {'long_name': 'identifier of selected reaches (REC)'}
attrs_FDC = {'description': 'FDC from hourly data', 'standard_name': 'FDC',
             'long_name': 'Flow Duration Curve using hourly data', 'units': 'meter3 second-1', 'valid_min': 0.0,
             'valid_max': 1e+20}
attrs_percentile = {'standard_name': 'Percentile', 'long_name': 'Percentile of exceedence', 'valid_min': 0.0,
                    'valid_max': 1.0, 'units': '[0..1]'}
attrs_year_removed = {'description': 'year removed from to construct the corresponding FDC'}


def calculate_probabilities(number_bins=0, start_prob=0.001, end_prob=.999):
    # Calculate the probabilities for the FDC
    if number_bins == 0:
        probabilities_fdc = np.array([0.0001, 0.0003, 0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                      0.6, 0.7, 0.8, 0.9, 0.95, 0.9950, 0.999, 0.9997, 0.9999])
    else:
        probabilities_fdc = np.linspace(start_prob, end_prob, number_bins)

    return probabilities_fdc


def calculate_FDC(probabilities_FDC: np.ndarray, flow_values: np.ndarray):
    """
    The actual calculation from the FDC from a list containing flow values. The values are sorted,
    and then the probabilities for each one are calculated using the Weibull plotting position. Finally,
    those values are sampled by either using a fixed set of probabilities (number_bins=0) or having a uniform
    of number_bins between start_prob and end_prob. The FDC values are finally log10'ed.

    Parameters
    ----------
    probabilities_FDC
    flow_values

    Returns
    -------

    """
    sorted_flow_values = np.sort(flow_values)[::-1]
    # Weibull plotting position
    probabilities_flow_values = np.array(list(range(len(sorted_flow_values)))) / (len(sorted_flow_values) + 1)

    # interpolate to the FDC values
    FDC = np.interp(probabilities_FDC, probabilities_flow_values, sorted_flow_values)
    return FDC


def leave_one_year_out(DA: xr.DataArray, year: int) -> np.ndarray:
    return DA.sel(time=~ (DA.time.dt.year == year)).values


def create_FDC_dict_sim(DS: xr.Dataset, number_of_exceedence=1001, output_variable_name: str = "Sim_FDC"):
    reaches = DS.rchid.values
    flow_variable_name = "mod_streamq"

    probabilities_FDC = calculate_probabilities(number_of_exceedence)

    print("Calculation of simulated FDC")
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

        values_FDC = calculate_FDC(probabilities_FDC, flow_values)

        data.append(values_FDC)

    FDC_dict = {
        'rchid': {'dims': ('nrch',), 'data': reaches, 'attrs': attrs_rchid},
        output_variable_name: {'dims': ('nrch', 'exceed'), 'data': data, 'attrs': attrs_FDC},
        'percentile': {'dims': ('exceed',), 'data': probabilities_FDC},
    }

    return FDC_dict


def create_FDC_dict_obs(DS: xr.Dataset, number_of_exceedence=1001,
                        output_variable_name: str = "Obs_FDC"):
    station_rchids = DS.station_rchid.values.astype(int)

    probabilities_FDC = calculate_probabilities(number_of_exceedence)

    data = []
    print("Calculation of observed FDC")

    good_stations = []
    for station_idx, station_rchid in enumerate(station_rchids):
        print(station_rchid)

        flow_values = select_valid_years(DS, station_idx)
        if len(flow_values) > 0:
            values_fdc = calculate_FDC(probabilities_FDC, flow_values)
            data.append(values_fdc)
            good_stations.append(station_rchid)
        else:
            print("Ignoring...")

    FDC_dict = {
        'station_rchid': {'dims': ('station',), 'data': good_stations, "attrs": attrs_station_rchid},
        output_variable_name: {'dims': ('station', 'exceed'), 'data': data, 'attrs': attrs_FDC},
        'percentile': {'dims': ('exceed',), 'data': probabilities_FDC, 'attrs': attrs_percentile},
    }

    return FDC_dict


def create_FDC_dict_cross_validation(DS: xr.Dataset, flow_variable_name: str = "mod_streamq",
                                     number_of_exceedence=1001, output_variable_name: str = "Sim_FDC",
                                     start_year: int = 1972, end_year: int = 2019):
    reaches = DS.rchid.values
    probabilities_FDC = calculate_probabilities(number_of_exceedence)
    if start_year < DS.time.dt.year.min().values:
        print("Start year is too small. Reverting with {}".format(DS.time.dt.year.min().values))
        start_year = DS.time.dt.year.min().values
    if end_year > DS.time.dt.year.max().values:
        print("End year is too big. Reverting with {}".format(DS.time.dt.year.min().values))
        end_year = DS.time.dt.year.max().values
    print("Calculation for Simulated FDC")
    print("Calculations for: ", start_year, end_year)

    data = []

    for year in range(start_year, end_year + 1):
        reduced_DS = DS.sel(time=~ (DS.time.dt.year == year))
        print(year)

        dimensions_flow_variable = len(DS.variables[flow_variable_name].dims)

        flow_values = np.ndarray([])
        if dimensions_flow_variable == 2:
            flow_values = reduced_DS.variables[flow_variable_name][:, :].values
        elif dimensions_flow_variable == 4:
            flow_values = reduced_DS.variables[flow_variable_name][:, :, 0, 0].values
        else:
            print("Flow values variable has a weird number of dimensions: ", dimensions_flow_variable)
            exit(1)

        values_FDC = np.apply_along_axis(lambda values: calculate_FDC(probabilities_FDC, values), axis=0,
                                         arr=flow_values)
        values_FDC = values_FDC.T
        print(values_FDC.shape)
        data.append(values_FDC)

    FDC_dict = {
        'rchid': {'dims': ('nrch',), 'data': reaches},
        output_variable_name: {'dims': ('year', 'nrch', 'exceed'), 'data': data, 'attrs': attrs_FDC},
        'percentile': {'dims': ('exceed',), 'data': probabilities_FDC, 'attrs': attrs_percentile},
        'year_removed': {'dims': ('year',), 'data': range(start_year, end_year + 1), 'attrs': attrs_year_removed}
    }
    return FDC_dict
