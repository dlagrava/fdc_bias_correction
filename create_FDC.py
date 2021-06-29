import FDC_calculation.FDC_calculation as FDC
import pandas as pd
import xarray as xr
import numpy as np
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("simulated_flows", type=str, )
    parser.add_argument("-observed_flows", type=str, default="")
    parser.add_argument("-cross_validation", action="store_true")
    parser.add_argument("-output_file_name", type=str, default="output.nc")
    parser.add_argument("-number_of_exceedence", type=int, default=0)
    parser.set_defaults(cross_validation=False)
    return parser.parse_args()

def main():
    args = parse_arguments()
    simulated_nc = args.simulated_flows
    observed_nc = args.observed_flows
    output_file_name = args.output_file_name
    cross_validation = args.cross_validation
    number_bins = args.number_of_exceedence

    simulated_DS = xr.open_dataset(simulated_nc)
    if cross_validation:
        FDC_dictionary = FDC.create_FDC_dict_cross_validation(simulated_DS, number_of_exceedence=number_bins,
                                                              start_year=2000, end_year=2010)
    else:
        FDC_dictionary = FDC.create_FDC_dict_sim(simulated_DS, number_of_exceedence=number_bins)

    if observed_nc != "":
        observed_DS = xr.open_dataset(observed_nc)
        FDC_dictionary_obs = FDC.create_FDC_dict_obs(observed_DS, number_of_exceedence=number_bins)
        FDC_dictionary.update(FDC_dictionary_obs)

    output_DS = xr.Dataset.from_dict(FDC_dictionary)
    output_DS.to_netcdf(output_file_name)


if __name__ == "__main__":
    main()
