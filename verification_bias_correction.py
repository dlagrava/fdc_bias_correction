import matplotlib.pyplot as plt
import scipy.stats
import xarray as xr
import numpy as np

import argparse
import os


def NSE(modelled_values, observed_values):
    """
    Nash-Sutcliffe Efficiency (NSE) as per formula found on several papers

    Parameters
    ----------
    modelled_values
    observed_values

    Returns
    -------

    """
    mean_obs = np.mean(observed_values)
    return 1 - np.sum((modelled_values - observed_values) ** 2) / np.sum((observed_values - mean_obs) ** 2)


def KGE(modelled_values, observed_values):
    """
    Kling-Gupta Efficiency (KGE) as per formula found on
    https://hess.copernicus.org/preprints/hess-2019-327/hess-2019-327.pdf

    Parameters
    ----------
    modelled_values
    observed_values

    Returns
    -------

    """
    mean_obs = np.mean(observed_values)
    mean_mod = np.mean(modelled_values)
    std_obs = np.std(observed_values)
    std_mod = np.std(modelled_values)
    pearson_correlation, _ = scipy.stats.pearsonr(modelled_values, observed_values)

    return 1 - np.sqrt(
        (pearson_correlation - 1.) ** 2 + (std_mod / std_obs - 1.) ** 2 + (mean_mod / mean_obs - 1.) ** 2)


def pBias(modelled_values, observed_values):
    """
    Percentual Bias (pBias) calculation. It expresses how we over or under-predict with the modelled values.

    Parameters
    ----------
    modelled_values
    observed_values

    Returns
    -------

    """
    return np.sum(observed_values - modelled_values) * 100. / sum(observed_values)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_nc")
    parser.add_argument("-output_dir", type=str, default=".")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    input_nc = args.input_nc
    output_dir = args.output_dir

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

    for reach_idx, reach_id in enumerate(all_reaches):

        for year in all_years:
            output_file_name = "{}/{}_{}".format(output_dir, reach_id, year)
            current_year_slice = slice('%i-01-01' % year, '%i-12-31' % year)
            reduced_input_ds = input_ds.sel(time=current_year_slice)

            simulated_flow = reduced_input_ds.raw_simulation[:, reach_idx].values
            bc_flow = reduced_input_ds.bias_corrected[:, reach_idx].values
            observed_flow = reduced_input_ds.river_flow_rate[:, reach_idx].values

            # Getting only values where we have observations
            valid_values_obs = observed_flow[np.argwhere(~np.isnan(observed_flow))].flatten()
            valid_values_sim = simulated_flow[np.argwhere(~np.isnan(observed_flow))].flatten()
            valid_values_bc_sim = bc_flow[np.argwhere(~np.isnan(observed_flow))].flatten()

            if len(valid_values_sim) < 100:
                print("Less than 100 hours of data for the observations for the year {}, ignoring.".format(year))
                continue

            # calculate yearly statistics for simulated and bc
            NSE_raw = NSE(valid_values_sim, valid_values_obs)
            NSE_bc = NSE(valid_values_bc_sim, valid_values_obs)
            NSE_log_sim = NSE(np.log10(valid_values_sim), np.log10(valid_values_obs))
            NSE_log_bc = NSE(np.log10(valid_values_bc_sim), np.log10(valid_values_obs))
            pBias_sim = pBias(valid_values_sim, valid_values_obs)
            pBias_bc = pBias(valid_values_bc_sim, valid_values_obs)
            KGE_sim = KGE(valid_values_sim, valid_values_obs)
            KGE_bc = KGE(valid_values_bc_sim, valid_values_obs)
            col_labels = ["", "Raw simulation", "Bias-corrected"]
            table_data = [["NSE", NSE_raw, NSE_bc], ["NSE (log)", NSE_log_sim, NSE_log_bc],
                          ["pBias", pBias_sim, pBias_bc], ["KGE", KGE_sim, KGE_bc]]

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
