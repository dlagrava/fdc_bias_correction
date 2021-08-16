import pandas as pd

from statistical_transformations.utils import get_common_reaches
from statistical_transformations.fdc_utils import convert_parameter_based_to_discrete

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import argparse
import os


def plot_all_reaches(input_ds: xr.Dataset, characteristics_df: pd.DataFrame,
                            distribution: str, output_dir: str = "."):
    """
    Plottign utility that will plot both the original discretized FDC and the values out of the parameter-based
    FDC

    Parameters
    ----------
    input_ds: xr.Dataset
        Input dataset, which should contian the variables Obs_FDC and Obs_FDC_<distribution>
    characteristics_df: pd.DataFrame
        Dataframe indexed by reach id and containing at least upstream (ie., upstream area)
    distribution: str
        A valid distribution. The parameter-based FDC were calculated for lognorm and genextreme
    output_dir: str
        Where to put all the plots.

    Returns
    -------
    None

    """
    common_reaches = get_common_reaches(input_ds)
    station_rchid = input_ds.station_rchid.values

    exceedance_values = input_ds.percentile.values

    for reach in common_reaches:
        idx_station_rchid = np.where(station_rchid == reach)[0][0]

        obs_fdc = input_ds.Obs_FDC[idx_station_rchid, :].values
        parameters_fdc = input_ds.variables["Obs_FDC_{}".format(distribution)][idx_station_rchid, :].values
        parameter_obs_fdc = convert_parameter_based_to_discrete(distribution, parameters_fdc, exceedance_values)
        upstream_area = characteristics_df.loc[reach, "uparea"]
        # Rescale to m3/s
        parameter_obs_fdc = upstream_area * 10. ** parameter_obs_fdc
        print(parameter_obs_fdc, obs_fdc)
        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.semilogy(exceedance_values, obs_fdc, label="Discrete FDC")
        plt.semilogy(exceedance_values, parameter_obs_fdc, label="Discretized FDC from parameter-based distribution")
        plt.legend()
        plt.title("Reach: {}".format(reach))

        plt.savefig("{}/comparison_discrete_parameter_FDC_{}.png".format(output_dir, reach))
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fdc_nc")
    parser.add_argument("properties_csv")
    parser.add_argument("distribution", choices=["lognorm", "genextreme"])
    parser.add_argument("-output_dir", help="Directory that will contain the results", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    input_fdc_nc = args.input_fdc_nc
    properties_csv = args.properties_csv
    output_dir = args.output_dir
    distribution = args.distribution

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    fdc_ds = xr.open_dataset(input_fdc_nc)
    characteristics_df = pd.read_csv(properties_csv, index_col="rchid")

    plot_all_reaches(fdc_ds, characteristics_df, distribution, output_dir=output_dir)


if __name__ == "__main__":
    main()
