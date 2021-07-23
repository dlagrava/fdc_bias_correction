from statistical_transformations.utils import get_common_reaches

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

from typing import List
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_nc", help="Input netcdf file that should have either quantiles or FDC")
    parser.add_argument("-output_dir", type=str, default=".", help="Place to store the images")
    args = parser.parse_args()

    return args


def generate_all_qq_plots(input_ds: xr.Dataset, common_reaches: List[int], output_dir):
    quantiles = input_ds.percentile.values
    rchid = input_ds.rchid.values
    station_rchid = input_ds.station_rchid.values

    for reach in common_reaches:
        idx_rchid = np.where(rchid == reach)[0][0]
        idx_station_rchid = np.where(station_rchid == reach)[0][0]

        sim_QQ = input_ds.Sim_Quantile[idx_rchid, :].values
        obs_QQ = input_ds.Obs_Quantile[idx_station_rchid, :].values

        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if obs_QQ.max() > -9999:
            plt.plot(sim_QQ, obs_QQ, marker=".", label="Actual QQ")
            plt.plot(obs_QQ, obs_QQ, linestyle='--', label="If Sim == Obs")
            plt.legend()
            plt.title("QQ plot for {} (sim vs. obs {} quantiles)".format(reach, len(quantiles)))
            plt.ylabel("Observed quantiles [cumecs]")
            plt.xlabel("Simulated quantiles [cumecs]")
            plt.savefig("{}/QQ_{}.png".format(output_dir, reach))
            plt.close()


def generate_all_fdc_plots(input_ds: xr.Dataset, common_reaches: List[int], output_dir):
    exceedence = input_ds.percentile.values
    rchid = input_ds.rchid.values
    station_rchid = input_ds.station_rchid.values

    for reach in common_reaches:
        idx_rchid = np.where(rchid == reach)[0][0]
        idx_station_rchid = np.where(station_rchid == reach)[0][0]
        if len(input_ds.Sim_FDC.dims) == 2:
            sim_fdc = input_ds.Sim_FDC[idx_rchid, :].values
        else:
            sim_fdc = input_ds.Sim_FDC[:, idx_rchid, :].values

        obs_fdc = input_ds.Obs_FDC[idx_station_rchid, :].values

        plt.figure(figsize=(12, 9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if len(input_ds.Sim_FDC.dims) == 2:
            plt.semilogy(exceedence, sim_fdc, label="Simulated FDC")
        else:
            years = len(input_ds.year_removed.values)
            for year_idx in range(years):
                plt.semilogy(exceedence, sim_fdc[year_idx, :],
                             label="Simulated FDC ({} removed)".format(input_ds.year_removed.values[year_idx]))

        if obs_fdc.max() > -9999:
            plt.semilogy(exceedence, obs_fdc, label="Observed FDC")
        plt.legend()
        plt.title(reach)

        plt.savefig("{}/FDC_{}.png".format(output_dir, reach))
        plt.close()


def main():
    args = parse_args()
    input_nc = args.input_nc
    output_dir = args.output_dir
    input_ds = xr.open_dataset(input_nc)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    common_reaches = get_common_reaches(input_ds)

    if "Sim_FDC" in input_ds.variables and "Obs_FDC" in input_ds.variables:
        generate_all_fdc_plots(input_ds, common_reaches, output_dir)
    elif "Sim_Quantile" in input_ds.variables and "Obs_Quantile" in input_ds.variables:
        generate_all_qq_plots(input_ds, common_reaches, output_dir)
    else:
        print("Nothing to plot on {}".format(input_nc))
        print("Input a netCDF4 file with either SimFDC and Obs_FDC or Sim_Quantile and Obs_Quantile variables")
        exit(0)


if __name__ == "__main__":
    main()
