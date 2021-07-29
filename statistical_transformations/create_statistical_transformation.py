import quantile_mapping_utils as QQ
import fdc_utils as FDC
import xarray as xr
import argparse

"""
Script to calculate the quantile mapping dataset for given simulations and observations.
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-simulated_flows", type=str, default="", help="netCDF file containing ")
    parser.add_argument("-observed_flows", type=str, default="")
    parser.add_argument("-output_file_name", type=str, default="output.nc")
    parser.add_argument("-number_of_values", type=int, default=1001,
                        help="Number of bins for the statistical transformation (exceedance or quantiles)")
    parser.add_argument("-method", choices=["FDC", "QM"], default="FDC")
    parser.add_argument("-cross_validation", action="store_true")
    parser.set_defaults(cross_validation=False)
    return parser.parse_args()


def main():
    args = parse_arguments()
    simulated_nc = args.simulated_flows
    observed_nc = args.observed_flows
    output_file_name = args.output_file_name
    number_of_values = args.number_of_values
    cross_validation = args.cross_validation
    method = args.method

    result_dictionary = {}
    if simulated_nc != "":
        print("simulations")
        simulated_ds = xr.open_dataset(simulated_nc)
        dictionary_sim = {}
        # Quantile mapping for simulation
        if method == "QM":
            dictionary_sim = QQ.create_quantile_dict_sim(simulated_ds, number_of_quantiles=number_of_values)
        # FDC for simulation
        elif method == "FDC":
            # FDC has the possibility of having cross-validation set
            if cross_validation:
                dictionary_sim = FDC.create_FDC_dict_cross_validation(simulated_ds,
                                                                      number_of_exceedence=number_of_values,
                                                                      start_year=2000, end_year=2010)
            else:
                dictionary_sim = FDC.create_FDC_dict_sim(simulated_ds, number_of_exceedence=number_of_values)
        # put the result
        result_dictionary.update(dictionary_sim)

    if observed_nc != "":
        print("Observations")
        dictionary_obs = {}
        observed_ds = xr.open_dataset(observed_nc)
        if method == "QM":
            dictionary_obs = QQ.create_quantile_dict_obs(observed_ds, number_of_quantiles=number_of_values)
        elif method == "FDC":
            dictionary_obs = FDC.create_FDC_dict_obs(observed_ds, number_of_exceedence=number_of_values)

        result_dictionary.update(dictionary_obs)

    if not result_dictionary:
        print("Nothing to write...")
        exit(0)

    output_ds = xr.Dataset.from_dict(result_dictionary)
    output_ds.attrs["statistical_transformation"] = method
    output_ds.to_netcdf(output_file_name)


if __name__ == "__main__":
    main()
