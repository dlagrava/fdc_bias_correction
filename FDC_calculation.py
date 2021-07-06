import xarray as xr
import numpy as np

attrs_station_rchid = {"long_name": "REC identifier for the stream reach on which station is located",
                       "valid_min": 0, "valid_max": 2000000000}
attrs_rchid = {'long_name': 'identifier of selected reaches (REC)'}
attrs_FDC = {'description': 'FDC from hourly data', 'standard_name': 'FDC',
             'long_name': 'Flow Duration Curve using hourly data', 'units': 'meter3 second-1', 'valid_min': 0.0,
             'valid_max': 1e+20}
attrs_percentile = {'standard_name': 'Percentile', 'long_name': 'Percentile of exceedence', 'valid_min': 0.0,
                    'valid_max': 1.0, 'units': '[0..1]'}
attrs_year_removed = {'description': 'year removed from to construct the corresponding FDC'}


def select_valid_years(input_flow_ds: xr.Dataset, station,
                       min_valid_years: int = 6, max_missing_days: int = 30,
                       flow_variable_name: str = "river_flow_rate") -> np.ndarray:
    """
    Take an xarray DataSet and select all the data from flow_variable. Calculate the missing data and
    make sure that we have at least min_valid_years (a complete year is missing less than max_missing_days of data).

    Parameters
    ----------
    input_flow_ds: xr.Dataset
        Input Dataset containing flow_variable
    station:
        The input station
    min_valid_years
    max_missing_days
    flow_variable_name

    Returns
    -------

    """
    valid_data = []
    valid_years = 0
    start_year = input_flow_ds.time.dt.year.min().values
    end_year = input_flow_ds.time.dt.year.max().values
    print(start_year, end_year)

    for year in np.arange(start_year, end_year + 1):
        year_slice = slice('%i-01-01' % year, '%i-12-31' % year)
        yearly_data = input_flow_ds.sel(time=year_slice)[flow_variable_name][:, station].to_masked_array()
        if len(yearly_data) - yearly_data.count() > max_missing_days:
            continue
        if np.count_nonzero(np.where(yearly_data >= 1e+35)) > 0:
            print("Wrong or missing values on the data for year", year)
            continue

        # Remove the 0 values?
        all_data = yearly_data.data[np.where(yearly_data.mask != True)]
        # TODO: check what to do with the 0 data
        all_data[all_data <= 0.] = np.min(all_data) * 0.01  # 1% of the all time minimal value
        # all_data = all_data[all_data > 0.] # remove all 0. data
        valid_data += list(all_data)
        valid_years += 1

    # print("Valid years: %i" % valid_years)
    if valid_years >= min_valid_years:
        return np.array(valid_data)

    # If we do not have enough data, we return an empty array
    return np.array([])


def calculate_probabilities(number_bins=0, start_prob=0.001, end_prob=.999):
    # Calculate the probabilities for the FDC
    if number_bins == 0:
        probabilities_FDC = np.array([0.0001, 0.0003, 0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                      0.6, 0.7, 0.8, 0.9, 0.95, 0.9950, 0.999, 0.9997, 0.9999])
    else:
        probabilities_FDC = np.linspace(start_prob, end_prob, number_bins)

    return probabilities_FDC


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

    for station_idx, station_rchid in enumerate(station_rchids):
        print(station_rchid)

        flow_values = select_valid_years(DS, station_idx)
        if len(flow_values) > 0:
            values_FDC = calculate_FDC(probabilities_FDC, flow_values)
        else:
            values_FDC = np.array([-9999.] * len(probabilities_FDC))

        data.append(values_FDC)

    FDC_dict = {
        'station_rchid': {'dims': ('station',), 'data': station_rchids, "attrs": attrs_station_rchid},
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

        values_FDC = np.apply_along_axis(lambda values: calculate_FDC(probabilities_FDC, values), axis=0, arr=flow_values)
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
