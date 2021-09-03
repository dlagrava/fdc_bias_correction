#! /bin/env python

from statistical_transformations.utils import add_observations_to_ds
import statistical_transformations.bias_correction_utils as bias_correction_utils

import argparse

import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation_nc", help="netCDF file containing the simulations")
    parser.add_argument("obs_stat_transf_nc",
                        help="netCDF file containing observed data for statistical transformations")
    parser.add_argument("sim_stat_transf_nc",
                        help="netCDF file containing simulated data for statistical transformations")
    parser.add_argument("-output_dir", default=".", help="Output directory for the results")
    parser.add_argument("-output_file_name", default="bias_corrected_fdc.nc")
    parser.add_argument("-observation_nc", default="")
    parser.add_argument("-FDC_sim_var", default="Sim_FDC", help="Name of the variable for the simulated FDC")
    parser.add_argument("-FDC_obs_var", default="Obs_FDC", help="Name of the variable for the observed FDC")
    parser.add_argument("-bias_correction_type", choices=["FDC_parameterbased", "FDC_crossvalidation",
                                                          "FDC_alltime", "FDC_seasonal", "QM_alltime"])
    parser.add_argument("-start_year", default=1975, help="Starting year for bias-correction")
    parser.add_argument("-end_year", default=2020, help="End year for bias-correction")
    return parser.parse_args()


def main():
    args = parse_args()
    simulation_nc = args.simulation_nc
    obs_stat_transf_nc = args.obs_stat_transf_nc
    sim_stat_transf_nc = args.sim_stat_transf_nc
    observation_nc = args.observation_nc
    output_dir = args.output_dir
    output_file_name = args.output_file_name
    bias_correction_type = args.bias_correction_type
    fdc_sim_var = args.FDC_sim_var
    fdc_obs_var = args.FDC_obs_var

    input_ds = xr.open_dataset(simulation_nc)
    obs_stat_transformation_ds = xr.open_dataset(obs_stat_transf_nc)
    sim_stat_transformation_ds = xr.open_dataset(sim_stat_transf_nc)

    start_year = args.start_year
    end_year = args.end_year

    if bias_correction_type == "FDC_crossvalidation":
        bias_corrected_dict = bias_correction_utils.bias_correction_cross_validation(input_ds, obs_stat_transformation_ds,
                                                                             sim_stat_transformation_ds,
                                                                             start_year=start_year, end_year=end_year)
        output_ds = xr.Dataset.from_dict(bias_corrected_dict)

    elif bias_correction_type == "FDC_seasonal":
        output_ds = bias_correction_utils.bias_correction_grouping(input_ds, obs_stat_transformation_ds, sim_stat_transformation_ds,
                                                                   statistical_transformation_type="FDC", grouping="season")

    elif bias_correction_type == "FDC_alltime":
        output_ds = bias_correction_utils.bias_correction_all_time(input_ds, obs_stat_transformation_ds,
                                                                             sim_stat_transformation_ds,
                                                                             start_year=start_year, end_year=end_year,
                                                                             fdc_sim_var=fdc_sim_var, fdc_obs_var=fdc_obs_var)

    else:
        assert False, "{} is not yet implemented.".format(bias_correction_type)

    # Add the observations as well, this helps in doing statistics
    if observation_nc != "":
        observation_ds = xr.open_dataset(observation_nc)
        output_ds = add_observations_to_ds(output_ds, observation_ds)

    output_ds.to_netcdf("{}/{}".format(output_dir, output_file_name))


if __name__ == "__main__":
    main()
