#! /bin/env python
import pandas as pd

from statistical_transformations.utils import NSE, KGE, pBias

import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import sklearn.metrics

import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_nc")
    parser.add_argument("-output_dir", type=str, default=".")
    parser.add_argument("-all_time_stats", action="store_true")
    parser.set_defaults(all_time_stats=False)
    args = parser.parse_args()

    return args

def calculate_performances(input_ds, reach_idx):
    """

    Parameters
    ----------
    input_ds
    reach_idx

    Returns
    -------

    """
    simulated_flow = input_ds.raw_simulation[:, reach_idx].values
    bc_flow = input_ds.bias_corrected[:, reach_idx].values
    observed_flow = input_ds.river_flow_rate[:, reach_idx].values

    # Getting only values where we have observations
    valid_values_obs = observed_flow[np.argwhere(~np.isnan(observed_flow))].flatten()
    valid_values_sim = simulated_flow[np.argwhere(~np.isnan(observed_flow))].flatten()
    valid_values_bc_sim = bc_flow[np.argwhere(~np.isnan(observed_flow))].flatten()

    if len(valid_values_sim) < 100:
        print("Less than 100 hours of data for the observations for the input, ignoring.")
        return {}

    # calculate yearly statistics for simulated and bc
    NSE_raw = NSE(valid_values_sim, valid_values_obs)
    NSE_bc = NSE(valid_values_bc_sim, valid_values_obs)
    NSE_log_raw = NSE(np.log10(valid_values_sim), np.log10(valid_values_obs))
    NSE_log_bc = NSE(np.log10(valid_values_bc_sim), np.log10(valid_values_obs))
    pBias_raw = pBias(valid_values_sim, valid_values_obs)
    pBias_bc = pBias(valid_values_bc_sim, valid_values_obs)
    KGE_raw = KGE(valid_values_sim, valid_values_obs)
    KGE_bc = KGE(valid_values_bc_sim, valid_values_obs)
    return {'NSE_raw': NSE_raw, 'NSE_bc': NSE_bc, "NSE_log_raw": NSE_log_raw, "NSE_log_bc": NSE_log_bc,
            'pBias_raw': pBias_raw, "pBias_bc": pBias_bc, "KGE_raw": KGE_raw, "KGE_bc": KGE_bc}

def main():
    args = parse_args()
    input_nc = args.input_nc
    output_dir = args.output_dir
    all_time_stats = args.all_time_stats

    if not os.path.isdir(output_dir):
        print("{} does not exist, creating...".format(output_dir))
        os.mkdir(output_dir)

    input_ds = xr.open_dataset(input_nc)

    # TODO: check that we have all the time-series we need to plot

    all_reaches = input_ds.rchid.values
    start_year = input_ds.time.dt.year.values.min()
    end_year = input_ds.time.dt.year.values.max()
    print(start_year, end_year)
    all_years = range(start_year, end_year + 1)

    if all_time_stats:
        selected_stats = ["NSE_raw", "NSE_bc", "NSE_log_raw", "NSE_log_bc", "KGE"]
        all_stats = []
        good_reaches = []
        for reach_idx, reach_id in enumerate(all_reaches):
            stats = calculate_performances(input_ds, reach_idx)
            if not stats:
                print("Ignoring: {}".format(reach_id))
                continue
            good_reaches.append(reach_id)
            all_stats.append([stats[i] for i in selected_stats])

        pd.DataFrame(data=all_stats, index=good_reaches, columns=selected_stats)
        pd.to_csv("{}/all_time_statistics.csv".format(output_dir))
        return

    # This part pertains yearly plots/stats
    for reach_idx, reach_id in enumerate(all_reaches):
        for year in all_years:
            output_file_name = "{}/{}_{}.png".format(output_dir, reach_id, year)
            current_year_slice = slice('%i-01-01' % year, '%i-12-31' % year)
            reduced_input_ds = input_ds.sel(time=current_year_slice)

            stats = calculate_performances(reduced_input_ds, reach_idx)

            col_labels = ["", "Raw simulation", "Bias-corrected"]
            table_data = [["NSE", stats["NSE_raw"], stats["NSE_bc"]], ["NSE (log)", stats["NSE_log_raw"], stats["NSE_log_bc"]],
                          ["pBias", stats["pBias_raw"], stats["pBias_bc"]], ["KGE", stats["KGE_raw"], stats["KGE_bc"]]]

            fig, axs = plt.subplots(2, 1, gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 8))
            print(fig, axs)
            axs[0].spines["top"].set_visible(False)
            axs[0].spines["right"].set_visible(False)
            fig.suptitle("Hydrographs for {} (year {})".format(reach_id, year))
            reduced_input_ds.raw_simulation[:, reach_idx].plot(label="Simulated flow", ax=axs[0])
            reduced_input_ds.bias_corrected[:, reach_idx].plot(label="bias-corrected flow", ax=axs[0])
            reduced_input_ds.river_flow_rate[:, reach_idx].plot(label="Observed flow", ax=axs[0])
            axs[0].legend()

            # table
            axs[1].axis('tight')
            axs[1].axis('off')
            the_table = axs[1].table(cellText=table_data, colLabels=col_labels, loc='center')
            plt.savefig(output_file_name)
            plt.close()


if __name__ == "__main__":
    main()
