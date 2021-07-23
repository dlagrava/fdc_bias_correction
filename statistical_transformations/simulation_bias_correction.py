from statistical_transformations.utils import add_observations_to_ds
import bias_correction_utils

import argparse

import xarray as xr
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation_nc", help="netCDF file containing the simulations")
    parser.add_argument("stat_transformation_nc", help="netCDF file containing simulated and observed statistical transformations")
    parser.add_argument("-output_dir", default=".", help="Output directory for the results")
    parser.add_argument("-output_file_name", default="bias_corrected_fdc.nc")
    parser.add_argument("-observation_nc", default="")
    parser.add_argument("-cross_validation", action="store_true")
    parser.set_defaults(cross_validation=False)
    return parser.parse_args()


def main():
    args = parse_args()
    simulation_nc = args.simulation_nc
    stat_transformation_nc = args.stat_transformation_nc
    observation_nc = args.observation_nc
    output_dir = args.output_dir
    output_file_name = args.output_file_name
    cross_validation = args.cross_validation

    input_ds = xr.open_dataset(simulation_nc)
    stat_transformation_ds = xr.open_dataset(stat_transformation_nc)

    start_year = 2000
    end_year = 2010

    if not cross_validation:
        bias_corrected_dict = bias_correction_utils.bias_correction_all_time(input_ds, stat_transformation_ds,
                                                                             start_year=start_year, end_year=end_year)
    else:
        bias_corrected_dict = bias_correction_utils.bias_correction_cross_validation(input_ds, stat_transformation_ds,
                                                                                     start_year=start_year,
                                                                                     end_year=end_year)

    output_ds = xr.Dataset.from_dict(bias_corrected_dict)

    if observation_nc != "":
        observation_ds = xr.open_dataset(observation_nc)
        output_ds = add_observations_to_ds(output_ds, observation_ds)

    output_ds.to_netcdf("{}/{}".format(output_dir, output_file_name))


if __name__ == "__main__":
    main()
