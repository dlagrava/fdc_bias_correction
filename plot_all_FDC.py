import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("fdc_nc")
    parser.add_argument("-output_dir", type=str, default=".")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    fdc_nc = args.fdc_nc
    output_dir = args.output_dir
    fdc_ds = xr.open_dataset(fdc_nc)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    rchid = fdc_ds.rchid.values
    station_rchid = np.array([])
    try:
        station_rchid = fdc_ds.station_rchid.values
    except ValueError:
        print("No observations are present")

    common_reaches = rchid
    if len(station_rchid) > 0:
        common_reaches = list(set(rchid).intersection(set(station_rchid)))
        prospect_common_reaches = np.array(common_reaches)
        if len(prospect_common_reaches) == 0:
            print("No common reaches!")
            exit(1)

    print(common_reaches)
    exceedence = fdc_ds.percentile.values

    for reach in common_reaches:
        idx_rchid = np.where(rchid == reach)[0][0]
        idx_station_rchid = np.where(station_rchid == reach)[0][0]
        if len(fdc_ds.Sim_FDC.dims) == 2:
            sim_fdc = fdc_ds.Sim_FDC[idx_rchid, :].values
        else:
            sim_fdc = fdc_ds.Sim_FDC[:, idx_rchid, :].values

        obs_fdc = fdc_ds.Obs_FDC[idx_station_rchid, :].values

        plt.figure(figsize=(12,9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if len(fdc_ds.Sim_FDC.dims) == 2:
            plt.semilogy(exceedence, sim_fdc, label="Simulated FDC")
        else:
            years = len(fdc_ds.year_removed.values)
            for year_idx in range(years):
                plt.semilogy(exceedence, sim_fdc[year_idx, :], label="Simulated FDC ({} removed)".format(fdc_ds.year_removed.values[year_idx]))

        if obs_fdc.max() > -9999:
            plt.semilogy(exceedence, obs_fdc, label="Observed FDC")
        plt.legend()
        plt.title(reach)

        plt.savefig("{}/FDC_{}.png".format(output_dir, reach))
        plt.close()


if __name__ == "__main__":
    main()
