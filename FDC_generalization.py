import xarray as xr
import pandas as pd
import numpy as np
import scipy.stats
from scipy.interpolate import interp1d

import argparse
import os


def resample_fdc(new_probs, original_probs, original_fdc):
    new_fdc = interp1d(original_probs, original_fdc,
                       kind='slinear', fill_value='extrapolate')(new_probs)
    return new_fdc


def calculate_parameters_fdc(distribution, original_fdc):
    dist = getattr(scipy.stats, distribution)
    params = dist.fit(original_fdc)
    return params


def prepare_fdc_for_training(fdc_ds: xr.Dataset, characteristics_df: pd.DataFrame,
                             method: str, selected_characteristics: list) -> pd.DataFrame:
    rchids_with_characteristics = characteristics_df.index
    rchids_with_fdc = fdc_ds.station_rchid.values

    # down-sampled probabilities
    probabilities_fdc = np.array([0.0001, 0.0003, 0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                  0.6, 0.7, 0.8, 0.9, 0.95, 0.9950, 0.999, 0.9997, 0.9999])

    distribution = None
    columns_result = probabilities_fdc
    fdc_type = method.split("_")[1]
    if fdc_type == "parameter":
        distribution = method.split("_")[2]
        columns_result = ["param0", "param1", "param2"]

    # get common reaches
    common_reaches = list(sorted(set(rchids_with_characteristics).intersection(rchids_with_fdc)))
    exceedence_rates = fdc_ds.exceed.values

    training_df = pd.DataFrame(index=common_reaches, columns=selected_characteristics + list(columns_result))

    for reach in common_reaches:
        characteristics_reach = characteristics_df.loc[reach][selected_characteristics]
        fdc_idx = np.where(fdc_ds.station_rchid.values == reach)[0][0]
        fdc_reach = fdc_ds.Obs_FDC[fdc_idx, :].values
        if fdc_type == "discrete":
            fdc = resample_fdc(probabilities_fdc, exceedence_rates, fdc_reach)
        else:
            assert distribution is not None, "Distribution should not be None"
            fdc = calculate_parameters_fdc(distribution, fdc_reach)

        # insert fdc values and characteristics in the resulting df
        training_df.loc[reach, selected_characteristics] = characteristics_reach
        training_df.loc[reach, columns_result] = fdc

    return training_df


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fdc_nc")
    parser.add_argument("properties_csv")
    parser.add_argument("method", choices=["rf_discrete", "mnn_discrete",
                                           "rf_parameter_genextreme", "rf_parameter_lognorm"])
    return parser.parse_args()


def main():
    args = parse_args()
    input_fdc_nc = args.input_fdc_nc
    properties_csv = args.properties_csv
    method = args.method

    input_fdc_ds = xr.open_dataset(input_fdc_nc)
    properties_df = pd.read_csv(properties_csv, index_col="rchid")

    filters = {'GoodSites': True, 'IsWeir': False}
    for k, v in filters.items():
        properties_df = properties_df.loc[properties_df[k] == v]

    # This is for NIWA data, we need to build CAMELS the same
    selected_characteristics = ["log10_elevation", "usRainDays10", "usPET", "usAnRainVar", "usParticleSize",
                                         "usAveSlope", "log10_uparea_m2", "CLIMATE"]

    joint_fdc_training = prepare_fdc_for_training(input_fdc_ds, properties_df, method, selected_characteristics)
    print(joint_fdc_training)
    categorical_characteristics = ["CLIMATE"]



if __name__ == "__main__":
    main()
