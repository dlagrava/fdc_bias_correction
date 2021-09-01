import xarray as xr
import numpy as np
import scipy.stats


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


def add_observations_to_ds(bc_ds: xr.Dataset, observation_ds: xr.Dataset) -> xr.Dataset:
    """
    Add observations to the bias-corrected data set. This allows for easier comparison.
    If there are no observations available, there will be a NAN value.

    Parameters
    ----------
    bc_ds: xr.Dataset
        Input data set containing at least simulations
    observation_ds: xr.Dataset
        Observed data set. Only reaches on the bias-corrected dataset will get a

    Returns
    -------
    The bias-corrected data set including observations for easier computation of statistics

    """
    bc_start_year = bc_ds.time.dt.year.min()
    bc_end_year = bc_ds.time.dt.year.max()
    bc_slice = slice('%i-01-01' % bc_start_year, '%i-12-31' % bc_end_year)
    print(bc_slice)
    reduced_observation_ds = observation_ds.sel(time=bc_slice)
    bc_reaches = bc_ds.rchid.values

    obs_values = np.ones_like(bc_ds.bias_corrected.values) * np.NAN

    for bc_rch_idx, bc_reach_id in enumerate(bc_reaches):
        print(bc_reach_id)
        try:
            obs_rch_idx = np.where(observation_ds.station_rchid.values == bc_reach_id)[0][0]
        except IndexError:
            print("No reach {} in observations, ignoring".format(bc_reach_id))
            continue

        obs_values[:, bc_rch_idx] = reduced_observation_ds.river_flow_rate[:, obs_rch_idx]

    bc_ds["river_flow_rate"] = bc_ds.bias_corrected.copy()
    bc_ds["river_flow_rate"].attrs.update(description="average flow", standard_name="river_flow_rate",
                                          long_name="observed_streamflow")
    bc_ds["river_flow_rate"][:, :] = obs_values

    return bc_ds


def get_common_reaches(input_ds: xr.Dataset):
    rchid = input_ds.rchid.values
    station_rchid = np.array([])
    try:
        station_rchid = input_ds.station_rchid.values
    except ValueError:
        print("No observations are present")

    common_reaches = rchid
    if len(station_rchid) > 0:
        common_reaches = list(set(rchid).intersection(set(station_rchid)))
        prospect_common_reaches = np.array(common_reaches)
        if len(prospect_common_reaches) == 0:
            print("No common reaches!")
            exit(0)
    else:
        print("No observed data to plot")
        exit(0)

    return common_reaches


def NSE(modelled_values, observed_values):
    """
    Nash-Sutcliffe Efficiency (NSE) as per formula found on several papers

    Parameters
    ----------
    modelled_values
    observed_values

    Returns
    -------

    """
    mean_obs = np.mean(observed_values)
    return 1 - np.sum((modelled_values - observed_values) ** 2) / np.sum((observed_values - mean_obs) ** 2)


def KGE(modelled_values, observed_values):
    """
    Kling-Gupta Efficiency (KGE) as per formula found on
    https://hess.copernicus.org/preprints/hess-2019-327/hess-2019-327.pdf

    Parameters
    ----------
    modelled_values
    observed_values

    Returns
    -------

    """
    mean_obs = np.mean(observed_values)
    mean_mod = np.mean(modelled_values)
    std_obs = np.std(observed_values)
    std_mod = np.std(modelled_values)
    pearson_correlation, _ = scipy.stats.pearsonr(modelled_values, observed_values)

    return 1 - np.sqrt(
        (pearson_correlation - 1.) ** 2 + (std_mod / std_obs - 1.) ** 2 + (mean_mod / mean_obs - 1.) ** 2)


def pBias(modelled_values, observed_values):
    """
    Percentual Bias (pBias) calculation. It expresses how we over or under-predict with the modelled values.

    Parameters
    ----------
    modelled_values
    observed_values

    Returns
    -------

    """
    return np.sum(observed_values - modelled_values) * 100. / sum(observed_values)
