from typing import Dict

import xarray as xr
import numpy as np
from scipy.interpolate import interp1d

"""
Utils to bias-correct time-series using statistical transformations. Two approaches are implemented:
- FDC bias-correction
- QM bias-correction

"""

attrs_rchid = {'long_name': 'identifier of selected reaches (REC)'}
attrs_bias_corrected = {'description': 'bias-corrected flow values', 'standard_name': 'bc_mod_streamq',
                        'long_name': 'Bias-corrected flow values', 'units': 'meter3 second-1', 'valid_min': 0.0,
                        'valid_max': 1e+20}
attrs_raw_simulation = {'description': 'simulation flow values', 'standard_name': 'mod_streamq',
                        'long_name': 'Bias-corrected flow values', 'units': 'meter3 second-1', 'valid_min': 0.0,
                        'valid_max': 1e+20}


def _remove_bias_flow_fdc(flow_values, simulated_fdc_probs, simulated_fdc, observed_fdc_probs, observed_fdc):
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
                                    kind='cubic', fill_value='extrapolate')(flow_values)
    # With that exceedance, compute the observed flow value
    bias_corrected_flow_values = interp1d(observed_fdc_probs, observed_fdc,
                                          kind='cubic', fill_value='extrapolate')(exceedance_simulated)
    return bias_corrected_flow_values


def _remove_bias_flow_qq(flow_values, simulated_quantiles, observed_quantiles):
    """
    Perform the algorithm for the bias-correction using a spline to interpolate
    between the simulated and observed quantiles (SSPLIN in Gudmundsson et al. 2012)

    Parameters
    ----------
    flow_values
    simulated_quantiles
    observed_quantiles

    Returns
    -------

    """
    bias_corrected_flow_values = interp1d(simulated_quantiles, observed_quantiles,
                                          kind='cubic', fill_value='extrapolate')(flow_values)
    return bias_corrected_flow_values


def get_statistical_transformation_available(stat_transformation_ds):
    """

    Parameters
    ----------
    stat_transformation_ds

    Returns
    -------

    """
    statistical_transformation_type = ""
    if "Sim_FDC" in stat_transformation_ds.variables and "Obs_FDC" in stat_transformation_ds.variables:
        statistical_transformation_type = "FDC"
    elif "Sim_Quantile" in stat_transformation_ds.variables and "Obs_Quantile" in stat_transformation_ds.variables:
        statistical_transformation_type = "QM"

    return statistical_transformation_type


def bias_correction_all_time(simulation_ds: xr.Dataset, obs_stat_transf_ds: xr.Dataset, sim_stat_transf_ds: xr.Dataset,
                             statistical_transformation_type: str = "FDC",
                             fdc_sim_var="Sim_FDC", fdc_obs_var="Obs_FDC"):
    """
    This function is a wrapper to select the available transformation in the simulation dataset. If the FDC
    is available, we will use Farmer et al. 2018. Otherwise, we use quantile mapping.

    Parameters
    ----------
    simulation_ds
    obs_stat_transf_ds
    sim_stat_transf_ds
    start_year
    end_year
    statistical_transformation_type
    fdc_sim_var
    fdc_obs_var

    Returns
    -------

    """
    flow_variable = "mod_streamq"

    simulation_rchid = simulation_ds.rchid.values

    time_coord = simulation_ds.time
    print(time_coord)
    simulation_ds["bc_mod_streamq"] = simulation_ds["mod_streamq"].copy()
    simulation_ds["bc_mod_streamq"].attrs['long_name'] = "timestep average streamflow (bias-corrected)"

    for i_reach, reach in enumerate(simulation_rchid):
        # check if we have valid FDC for simulation and observation
        try:
            simulated_fdc_idx = np.where(sim_stat_transf_ds.rchid.values == reach)[0][0]
            observed_fdc_idx = np.where(obs_stat_transf_ds.station_rchid.values == reach)[0][0]
            print(np.where(sim_stat_transf_ds.rchid.values == reach))
            print(np.where(obs_stat_transf_ds.station_rchid.values == reach))
        except IndexError as error:
            print("Error getting simulated/observed FDC, ignoring {}".format(reach))
            print(error)
            # Set the values to _fillValue
            simulation_ds["bc_mod_streamq"][:, i_reach, 0, 0] = -9999.
            continue

        flow_values = simulation_ds.variables[flow_variable][:, simulated_fdc_idx, 0, 0].values
        bias_corrected_values = np.array([])
        # Choose the correct functions according to the transformation type
        if statistical_transformation_type == "FDC":
            simulated_percentiles = sim_stat_transf_ds.variables["percentile"][:].values
            observed_percentiles = obs_stat_transf_ds.variables["percentile"][:].values
            simulated_fdc = sim_stat_transf_ds.variables[fdc_sim_var][simulated_fdc_idx, :].values
            observed_fdc = obs_stat_transf_ds.variables[fdc_obs_var][observed_fdc_idx, :].values
            bias_corrected_values = _remove_bias_flow_fdc(flow_values, simulated_percentiles, simulated_fdc,
                                                          observed_percentiles,
                                                          observed_fdc)

        if statistical_transformation_type == "QM":
            simulated_quantiles = sim_stat_transf_ds.variables["Sim_Quantile"][simulated_fdc_idx, :].values
            observed_quantiles = obs_stat_transf_ds.variables["Obs_Quantile"][observed_fdc_idx, :].values
            bias_corrected_values = _remove_bias_flow_qq(flow_values, simulated_quantiles, observed_quantiles)

        # TODO: this should happen for all nrun and nens as well to be perfectly correct
        simulation_ds["bc_mod_streamq"][:, i_reach, 0, 0] = bias_corrected_values

    return simulation_ds


def bias_correction_grouping(simulation_ds: xr.Dataset, obs_stat_transf_ds: xr.Dataset, sim_stat_transf_ds: xr.Dataset,
                             start_year=1972, end_year=2019, statistical_transformation_type="FDC", grouping="season"):
    """
    This function is a wrapper to select the available transformation in the simulation dataset. If the FDC
    is selected, we will use Farmer et al. 2018. Otherwise, we use quantile mapping.

    Parameters
    ----------
    simulation_ds
    stat_transformation_ds
    start_year
    end_year

    Returns
    -------

    """
    flow_variable = "mod_streamq"
    bias_corrected_variable = "bc_streamq"

    simulation_rchid = simulation_ds.rchid.values

    time_coord = simulation_ds.time
    print(time_coord)

    simulation_ds[bias_corrected_variable] = simulation_ds.variables[flow_variable].copy()

    for reach in simulation_rchid:
        # check if we have valid FDC for simulation and observation
        try:
            print(np.where(sim_stat_transf_ds.rchid.values == reach))
            print(np.where(obs_stat_transf_ds.station_rchid.values == reach))
            simulated_fdc_idx = np.where(sim_stat_transf_ds.rchid.values == reach)[0][0]
            observed_fdc_idx = np.where(obs_stat_transf_ds.station_rchid.values == reach)[0][0]
        except IndexError as error:
            print("Error getting simulated/observed FDC, ignoring {}".format(reach))
            print(error)
            continue

        grouping_idx = simulation_ds.groupby('time.{}'.format(grouping))
        bias_corrected_values = np.array([])
        # Choose the correct functions according to the transformation type
        if statistical_transformation_type == "FDC":
            simulated_percentiles = sim_stat_transf_ds.variables["percentile"][:].values
            observed_percentiles = obs_stat_transf_ds.variables["percentile"][:].values

            grouped_bias_corrected_values = []
            for group in grouping_idx:
                simulated_fdc = sim_stat_transf_ds.variables["Sim_FDC_{}".format(grouping)][simulated_fdc_idx, :,
                                group].values
                observed_fdc = obs_stat_transf_ds.variables["Obs_FDC_{}".format(grouping)][observed_fdc_idx, :,
                               group].values
                grouped_simulated_values = simulation_ds[flow_variable][:, simulation_rchid, 0, 0].isel(
                    time=grouping_idx.groups[group])
                bias_corrected_simulated_values = _remove_bias_flow_fdc(grouped_simulated_values, simulated_percentiles,
                                                                        simulated_fdc, observed_percentiles,
                                                                        observed_fdc)
                grouped_simulated_values.values = bias_corrected_simulated_values
                grouped_bias_corrected_values.append(grouped_simulated_values)

            assert len(bias_corrected_values) > 0., "Bias corrected values have 0 values, this is a problem"
            joint_seasonal_values = xr.concat(grouped_bias_corrected_values, dim='time')
            joint_seasonal_values = joint_seasonal_values.sortby('time')
            simulation_ds[bias_corrected_variable][:, simulation_rchid, 0, 0] = joint_seasonal_values
        else:
            assert False, "We have only implemented grouping for FDC."

    return simulation_ds


def bias_correction_cross_validation(simulation_ds: xr.Dataset, obs_stat_transformation_ds: xr.Dataset,
                                     sim_stat_transformation_ds: xr.Dataset, start_year=1972,
                                     end_year=2019) -> Dict:
    """
    This function assumes that the statistical transformation is applied on values that have been prepared for
    cross-validation. To bias-correct a certain year, the statistics (FDC or QM) are computed on all time
    data minus that year.

    TODO: Not implemented

    Parameters
    ----------
    simulation_ds
    stat_transformation_ds
    start_year
    end_year

    Returns
    -------

    """

    assert False, "Not implemented yet!"

    return {}
