import bias_correction.FDC_bias_correction

import argparse
import os

import numpy as np
import xarray as xr


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_nc_file", help="netCDF file containing the simulations")
    parser.add_argument("fdc_nc_file", help="netCDF file containing simulated and observed FDC")
    parser.add_argument("-output_dir", default=".")
    parser.add_argument("-output_file_name", default="bias_corrected_fdc.nc")
    return parser.parse_args()


def main():
    args = parse_args()
    input_nc = args.input_nc_file
    fdc_nc = args.fdc_nc_file
    output_dir = args.output_dir
    output_file_name = args.output_file_name

    input_ds = xr.open_dataset(input_nc)
    fdc_ds = xr.open_dataset(fdc_nc)

    bias_corrected_dict = bias_correction.FDC_bias_correction.bias_correction_all_time(input_ds, fdc_ds,
                                                                                       start_year=1990, end_year=2000)

    output_ds = xr.Dataset.from_dict(bias_corrected_dict)
    output_ds.to_netcdf("{}/{}".format(output_dir, output_file_name))


if __name__ == "__main__":
    main()
