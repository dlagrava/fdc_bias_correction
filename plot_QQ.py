import matplotlib.pyplot as plt
import xarray as xr
import numpy as np

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("qq_nc")
    parser.add_argument("-output_dir", type=str, default=".")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    qq_nc = args.qq_nc
    output_dir = args.output_dir
    qq_ds = xr.open_dataset(qq_nc)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    rchid = qq_ds.rchid.values
    station_rchid = np.array([])
    try:
        station_rchid = qq_ds.station_rchid.values
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
    quantiles = qq_ds.percentile.values

    for reach in common_reaches:
        idx_rchid = np.where(rchid == reach)[0][0]
        idx_station_rchid = np.where(station_rchid == reach)[0][0]

        sim_QQ = qq_ds.Sim_Quantile[idx_rchid, :].values
        obs_QQ = qq_ds.Obs_Quantile[idx_station_rchid, :].values

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


if __name__ == "__main__":
    main()
