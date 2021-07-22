import xarray as xr
import numpy as np


def select_valid_years(input_flow_ds: xr.Dataset, station,
                       min_valid_years: int = 6, max_missing_hours: int = 30 * 24,
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
    max_missing_hours
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
        if len(yearly_data) - yearly_data.count() > max_missing_hours:
            continue
        if np.count_nonzero(np.where(yearly_data >= 1e+35)) > 0:
            print("Wrong or missing values on the data for year", year)
            continue

        # Remove all non-valid numbers
        all_data = yearly_data.data[np.where(yearly_data.mask != True)]

        if np.max(all_data) == 0.:
            print("Year full of 0s: ", len(all_data))
            print(all_data)
            continue

        # Remove the 0 values?
        # all_data = all_data[all_data > 0.] # remove all 0. data
        valid_data += list(all_data)
        valid_years += 1

    valid_data = np.array(valid_data)

    if valid_years >= min_valid_years:
        # Changing 0s to some valid value for logs
        valid_data[valid_data <= 0.] = np.min(valid_data[valid_data > 0.]) * 0.01  # 1% of the all time minimal value
        return valid_data

    # If we do not have enough data, we return an empty array
    return np.array([])
