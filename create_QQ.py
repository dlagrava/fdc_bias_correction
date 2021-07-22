import QQ_calculation as QQ
import xarray as xr
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-simulated_flows", type=str, default="")
    parser.add_argument("-observed_flows", type=str, default="")
    parser.add_argument("-output_file_name", type=str, default="output.nc")
    parser.add_argument("-number_of_quantile", type=int, default=1000)
    parser.set_defaults(cross_validation=False)
    return parser.parse_args()


def main():
    args = parse_arguments()
    simulated_nc = args.simulated_flows
    observed_nc = args.observed_flows
    output_file_name = args.output_file_name
    number_bins = args.number_of_quantile

    print("HERE")
    quantile_dictionary = {}
    if simulated_nc != "":
        print("QQ simulated")
        simulated_DS = xr.open_dataset(simulated_nc)
        quantile_dictionary_sim = QQ.create_quantile_dict_sim(simulated_DS, number_of_quantiles=number_bins)
        quantile_dictionary.update(quantile_dictionary_sim)

    if observed_nc != "":
        print("QQ observed")
        observed_DS = xr.open_dataset(observed_nc)
        quantile_dictionary_obs = QQ.create_quantile_dict_obs(observed_DS, number_of_quantiles=number_bins)
        quantile_dictionary.update(quantile_dictionary_obs)

    if not quantile_dictionary:
        print("Nothing to write...")
        exit(0)

    output_DS = xr.Dataset.from_dict(quantile_dictionary)
    output_DS.to_netcdf(output_file_name)




if __name__ == "__main__":
    main()
