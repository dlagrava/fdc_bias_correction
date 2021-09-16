import pandas as pd

from .utils import select_valid_years

import xarray as xr
import numpy as np
import scipy.stats

from typing import Dict

attrs_station_rchid = {"long_name": "REC identifier for the stream reach on which station is located",
                       "valid_min": 0, "valid_max": 2000000000}
attrs_rchid = {'long_name': 'identifier of selected reaches (REC)'}
attrs_FDC = {'description': 'FDC from hourly data', 'standard_name': 'FDC',
             'long_name': 'Flow Duration Curve using hourly data', 'units': 'meter3 second-1', 'valid_min': 0.0,
             'valid_max': 1e+20}
attrs_percentile = {'standard_name': 'Percentile', 'long_name': 'Percentile of exceedence', 'valid_min': 0.0,
                    'valid_max': 1.0, 'units': '[0..1]'}
attrs_year_removed = {'description': 'year removed from to construct the corresponding FDC'}


def calculate_probabilities(number_bins=0, start_prob=0.0, end_prob=1.0):
    # Calculate the probabilities for the FDC
    if number_bins == 0:
        probabilities_fdc = np.array([0.0001, 0.0003, 0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                      0.6, 0.7, 0.8, 0.9, 0.95, 0.9950, 0.999, 0.9997, 0.9999])
    else:
        probabilities_fdc = np.linspace(start_prob, end_prob, number_bins)

    return probabilities_fdc


def convert_parameter_based_to_discrete(distribution_name: str, parameters: np.ndarray, probabilities: np.ndarray):
    """


    Parameters
    ----------
    distribution_name
    parameters
    probabilities

    Returns
    -------

    """
    assert len(parameters) == 3, "We only support 3 parameter distributions yet."
    distribution = getattr(scipy.stats, distribution_name)
    rv = distribution(parameters[0], parameters[1], parameters[2])
    fdc = rv.ppf(1. - probabilities)

    return fdc


def calculate_FDC(probabilities_FDC: np.ndarray, flow_values: np.ndarray) -> np.ndarray:
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
    np.ndarray

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
    skip_initial_values = 2 * 365 * 24 # Remove the first 2 years of any simulation

    probabilities_FDC = calculate_probabilities(number_of_exceedence)

    print("Calculation of simulated FDC")
    data = []
    dimensions_flow_variable = len(DS.variables[flow_variable_name].dims)
    for idx, reach_id in enumerate(reaches):
        print(reach_id)

        flow_values = np.ndarray([])
        # The following test was added to possibly deal with CAMELS and CAMELS-UK data
        if dimensions_flow_variable == 2:
            flow_values = DS.variables[flow_variable_name][skip_initial_values:, idx].values
        elif dimensions_flow_variable == 4:
            flow_values = DS.variables[flow_variable_name][skip_initial_values:, idx, 0, 0].values
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


def create_FDC_dict_obs(DS: xr.Dataset, number_of_exceedence=1001, output_variable_name: str = "Obs_FDC",
                        characteristics=""):
    """


    Parameters
    ----------
    characteristics
    distribution
    DS
    number_of_exceedence
    output_variable_name
    distribution

    Returns
    -------

    """
    station_rchids = DS.station_rchid.values.astype(int)
    parameter_based_distributions = ["lognorm", "genextreme"]
    probabilities_fdc = calculate_probabilities(number_of_exceedence)

    characteristics_df = None
    if characteristics != "":
        characteristics_df = pd.read_csv(characteristics, index_col="rchid")

    data = []
    distributions_parameters = {dist: [] for dist in parameter_based_distributions}
    print("Calculation of observed FDC")

    good_stations = []
    for station_idx, station_rchid in enumerate(station_rchids):
        print(station_rchid)

        flow_values = select_valid_years(DS, station_idx)
        if len(flow_values) > 0:
            values_fdc = calculate_FDC(probabilities_fdc, flow_values)

            if characteristics_df is not None:
                try:
                    catchment_uparea = characteristics_df.loc[station_rchid, "uparea"]
                except KeyError:
                    print("Area for reach not available, ignoring it")
                    continue

                flow_values = np.log10(flow_values / catchment_uparea)
                for distribution in distributions_parameters:
                    dist = getattr(scipy.stats, distribution)
                    params = dist.fit(flow_values)
                    distributions_parameters[distribution].append(params)

            data.append(values_fdc)
            good_stations.append(station_rchid)
        else:
            print("Ignoring...")

    FDC_dict = {
        'station_rchid': {'dims': ('station',), 'data': good_stations, "attrs": attrs_station_rchid},
        output_variable_name: {'dims': ('station', 'exceed'), 'data': data, 'attrs': attrs_FDC},
        'percentile': {'dims': ('exceed',), 'data': probabilities_fdc, 'attrs': attrs_percentile},
    }

    for distribution in parameter_based_distributions:
        FDC_dict["{}_{}".format(output_variable_name, distribution)] = {
            'dims': ('station', 'parameters'), 'data': distributions_parameters[distribution], 'attrs': {
                "description": '{} parameters to fit log10 flow data divided by catchment area'.format(distribution),
                "units": "log10(m3/s/km2)",
                "name": distribution
            }
        }

    return FDC_dict


def create_FDC_dict_cross_validation(DS: xr.Dataset, flow_variable_name: str = "mod_streamq",
                                     number_of_exceedence=1001, output_variable_name: str = "Sim_FDC",
                                     start_year: int = 1972, end_year: int = 2019):
    """

    Parameters
    ----------
    DS
    flow_variable_name
    number_of_exceedence
    output_variable_name
    start_year
    end_year

    Returns
    -------

    """
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


def create_FDC_dict_sim_seasonal(simulated_ds: xr.Dataset, number_of_exceedence=1001,
                                 output_variable_name: str = "Sim_FDC", characteristics: str = "") -> Dict:
    station_rchids = simulated_ds.rchid.values.astype(int)
    characteristics_df = None
    if characteristics != "":
        characteristics_df = pd.read_csv(characteristics, index_col="rchid")

    probabilities_fdc = calculate_probabilities(number_of_exceedence)
    print("Calculation of simulated FDC (Seasonal)")
    seasonal_idx = simulated_ds.groupby('time.season')
    print(seasonal_idx)
    flow_variable_name = "mod_streamq"

    available_reaches = simulated_ds.coords["nrch"]

    data = np.zeros((len(available_reaches), len(probabilities_fdc), 4))
    for i_rchid, rchid in enumerate(available_reaches):
        reach_id = int(station_rchids[rchid])
        print(reach_id)

        flow_values = simulated_ds.variables[flow_variable_name][:, i_rchid, 0, 0].values
        values_FDC = calculate_FDC(probabilities_fdc, flow_values)

        # Divide the data seasonally
        for i_season, season in enumerate(seasonal_idx.groups):
            all_values_season = simulated_ds[flow_variable_name][:, rchid].isel(
                time=seasonal_idx.groups[season]).to_masked_array()
            all_values_season = all_values_season.data[np.where(all_values_season.mask != True)]

            # Construct the seasonal FDC
            values_fdc = calculate_FDC(probabilities_fdc, all_values_season)
            data[i_rchid, :, i_season] = values_fdc

    FDC_dict = {
        'rchid': {'dims': ('station',), 'data': station_rchids, 'attrs': attrs_station_rchid},
        output_variable_name: {'dims': ('station', 'exceed', 'season'), 'data': data, 'attrs': attrs_FDC},
        'percentile': {'dims': ('exceed',), 'data': probabilities_fdc, 'attrs': attrs_percentile},
        'seasons': {'dims': ('season',), 'data': ['DJF', 'MAM', 'JJA', 'SON'], 'attrs': {}}
    }

    return FDC_dict


def create_FDC_dict_sim_monthly(simulated_ds, number_of_exceedence) -> Dict:
    return {}


def create_FDC_dict_obs_seasonal(observed_ds: xr.Dataset, number_of_exceedence=1001,
                                 output_variable_name: str = "Obs_FDC_seasonal",
                                 characteristics="") -> Dict:
    station_rchids = observed_ds.station_rchid.values.astype(int)
    characteristics_df = None
    if characteristics != "":
        characteristics_df = pd.read_csv(characteristics, index_col="rchid")

    probabilities_fdc = calculate_probabilities(number_of_exceedence)
    print("Calculation of observed FDC (Seasonal)")
    seasonal_idx = observed_ds.groupby('time.season')
    print(seasonal_idx)
    selected_variable = "river_flow_rate"
    available_stations = observed_ds.coords["station"]

    data = np.zeros((len(available_stations), len(probabilities_fdc), 4))
    for i_station, station in enumerate(available_stations):
        reach_id = int(station_rchids[station])
        print(reach_id)
        # Check if the site passes the check for minimal data content
        flow_values = select_valid_years(observed_ds, station)
        if len(flow_values) == 0:
            print("{}: not enough values to construct the FDC. Ignoring".format(reach_id))
            data[i_station, :, :] = np.nan
            continue

        # Divide the data seasonally
        for i_season, season in enumerate(seasonal_idx.groups):
            all_values_season = observed_ds[selected_variable][:, station].isel(
                time=seasonal_idx.groups[season]).to_masked_array()
            all_values_season = all_values_season.data[np.where(all_values_season.mask != True)]

            # Construct the seasonal FDC
            values_fdc = calculate_FDC(probabilities_fdc, all_values_season)
            data[i_station, :, i_season] = values_fdc

    FDC_dict = {
        'rchid': {'dims': ('station',), 'data': station_rchids, 'attrs': attrs_station_rchid},
        output_variable_name: {'dims': ('station', 'exceed', 'season'), 'data': data, 'attrs': attrs_FDC},
        'percentile': {'dims': ('exceed',), 'data': probabilities_fdc, 'attrs': attrs_percentile},
        'seasons': {'dims': ('season',), 'data': ['DJF', 'MAM', 'JJA', 'SON'], 'attrs': {}}
    }

    return FDC_dict


def create_FDC_dict_obs_monthly(observed_ds, number_of_exceedence, characteristics) -> Dict:
    return {}
