import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

attrs_rchid = {'long_name': 'identifier of selected reaches (REC)'}
attrs_bias_corrected = {'description': 'bias-corrected flow values', 'standard_name': 'bc_mod_streamq',
                        'long_name': 'Bias-corrected flow values', 'units': 'meter3 second-1', 'valid_min': 0.0,
                        'valid_max': 1e+20}
attrs_raw_simulation = {'description': 'simulation flow values', 'standard_name': 'mod_streamq',
                        'long_name': 'Bias-corrected flow values', 'units': 'meter3 second-1', 'valid_min': 0.0,
                        'valid_max': 1e+20}


# plt.plot(flow_values, label="Simulated")
# plt.plot(bias_corrected_values, label="Bias-corrected")
# plt.title("Simulated vs. Bias-corrected for {}".format(reach))
# plt.legend()
# plt.savefig("{}/{}_bias_correction.png".format(output_dir, reach))
# plt.close()

def remove_bias_flow(flow_values, simulated_fdc_probs, simulated_fdc, observed_fdc_probs, observed_fdc):
    """
    Perform the algorithm for the bias-correction in 2 vectorized operations.

    Parameters
    ----------
    flow_values
    simulated_fdc_probs
    simulated_fdc
    observed_fdc_probs
    observed_fdc

    Returns
    -------

    """
    # Get the probability of flow_value on the simulation

    exceedance_simulated = interp1d(simulated_fdc[::-1], simulated_fdc_probs[::-1],
                                    kind='slinear', fill_value='extrapolate')(flow_values)
    # With that exceedance, compute the observed flow value
    bias_corrected_flow_values = interp1d(observed_fdc_probs, observed_fdc,
                                          kind='slinear', fill_value='extrapolate')(exceedance_simulated)
    return bias_corrected_flow_values


def bias_correction_all_time(simulation_ds: xr.Dataset, fdc_ds: xr.Dataset,
                             start_year=1972, end_year=2019, output_dir="."):
    flow_variable = "mod_streamq"
    time_slice = slice('%i-01-01' % start_year, '%i-12-31' % end_year)
    reduced_ds = simulation_ds.sel(time=time_slice)

    simulation_rchid = simulation_ds.rchid.values
    # Containers for output
    bias_corrected_reaches = []
    raw_values = []
    bias_corrected = []

    time_coord = reduced_ds.time
    print(time_coord)

    for reach in simulation_rchid:
        # check if we have valid FDC for simulation and observation
        try:
            print(np.where(fdc_ds.rchid.values == reach))
            print(np.where(fdc_ds.station_rchid.values == reach))
            simulated_fdc_idx = np.where(fdc_ds.rchid.values == reach)[0][0]
            observed_fdc_idx = np.where(fdc_ds.station_rchid.values == reach)[0][0]
        except IndexError as error:
            print("Error getting simulated/observed FDC, ignoring {}".format(reach))
            print(error)
            continue

        flow_values = reduced_ds.variables[flow_variable][:, simulated_fdc_idx, 0, 0].values
        percentiles = fdc_ds.variables["percentile"][:].values
        simulated_fdc = fdc_ds.variables["Sim_FDC"][simulated_fdc_idx, :].values
        observed_fdc = fdc_ds.variables["Obs_FDC"][observed_fdc_idx, :].values
        bias_corrected_values = remove_bias_flow(flow_values, percentiles, simulated_fdc, percentiles, observed_fdc)

        bias_corrected_reaches.append(reach)
        raw_values.append(flow_values)
        bias_corrected.append(bias_corrected_values)

    raw_values = np.array(raw_values).T
    bias_corrected = np.array(bias_corrected).T

    bias_corrected_dict = {
        'rchid': {'dims': ('nrch',), 'data': bias_corrected_reaches, 'attrs': attrs_rchid},
        'bias_corrected': {'dims': ('time', 'nrch'), 'data': bias_corrected, 'attrs': attrs_bias_corrected},
        'raw_simulation': {'dims': ('time', 'nrch'), 'data': raw_values, 'attrs': attrs_raw_simulation},
        'time': {'dims': ('time',), 'data': time_coord}
    }

    return bias_corrected_dict


def bias_correction_cross_validation(simulation_ds: xr.Dataset, fdc_ds: xr.Dataset, start_year=1972, end_year=2019):
    flow_variable = "mod_streamq"
    time_slice = slice('%i-01-01' % start_year, '%i-12-31' % end_year)
    reduced_ds = simulation_ds.sel(time=time_slice)

    # check that the desired time range is available
    assert start_year >= fdc_ds.year_removed.min(), "starting year is smaller than available dates."
    assert end_year <= fdc_ds.year_removed.max(), "end year is bigger than availabe dates."

    simulation_rchid = simulation_ds.rchid.values
    # Containers for output
    bias_corrected_reaches = []
    raw_values = []
    bias_corrected = []

    time_coord = reduced_ds.time
    print(time_coord)

    for reach in simulation_rchid:
        flow_values = np.array([])
        bias_corrected_values = np.array([])

        # check if we have valid FDC for simulation and observation
        try:
            print(np.where(fdc_ds.rchid.values == reach))
            print(np.where(fdc_ds.station_rchid.values == reach))
            simulated_fdc_idx = np.where(fdc_ds.rchid.values == reach)[0][0]
            observed_fdc_idx = np.where(fdc_ds.station_rchid.values == reach)[0][0]
        except IndexError as error:
            print("Error getting simulated/observed FDC, ignoring {}".format(reach))
            print(error)
            continue

        # Loop over each year and fetch the corresponding FDC
        print("Going from {} to {}".format(start_year, end_year))
        for current_year in range(start_year, end_year + 1):
            year_removed_idx = np.where(fdc_ds.year_removed.values == current_year)[0][0]
            year_slice = slice('%i-01-01' % current_year, '%i-12-31' % current_year)
            year_flow = reduced_ds.sel(time=year_slice).variables[flow_variable]
            year_flow_values = year_flow[:, simulated_fdc_idx, 0, 0].values
            percentiles = fdc_ds.variables["percentile"][:].values
            simulated_fdc = fdc_ds.variables["Sim_FDC"][year_removed_idx, simulated_fdc_idx, :].values
            observed_fdc = fdc_ds.variables["Obs_FDC"][observed_fdc_idx, :].values
            year_bias_corrected_values = remove_bias_flow(year_flow_values, percentiles, simulated_fdc, percentiles,
                                                          observed_fdc)

            flow_values = np.append(flow_values, year_flow_values)
            bias_corrected_values = np.append(bias_corrected_values, year_bias_corrected_values)

        # plt.plot(flow_values, label="Simulated")
        # plt.plot(bias_corrected_values, label="Bias-corrected")
        # plt.title("Simulated vs. Bias-corrected for {}".format(reach))
        # plt.legend()
        # plt.show()
        # plt.close()

        bias_corrected_reaches.append(reach)
        raw_values.append(flow_values)
        bias_corrected.append(bias_corrected_values)

    raw_values = np.array(raw_values).T
    bias_corrected = np.array(bias_corrected).T

    bias_corrected_dict = {
        'rchid': {'dims': ('nrch',), 'data': bias_corrected_reaches, 'attrs': attrs_rchid},
        'bias_corrected': {'dims': ('time', 'nrch'), 'data': bias_corrected, 'attrs': attrs_bias_corrected},
        'raw_simulation': {'dims': ('time', 'nrch'), 'data': raw_values, 'attrs': attrs_raw_simulation},
        'time': {'dims': ('time',), 'data': time_coord}
    }

    return bias_corrected_dict
