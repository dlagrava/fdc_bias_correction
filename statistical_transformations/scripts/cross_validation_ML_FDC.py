#! /bin/env python

import statistical_transformations.regression_models as regression_models

import xarray as xr
import pandas as pd
import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
import sklearn.model_selection
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import argparse
from typing import Sequence, Tuple, Any


def resample_fdc(new_probs: np.ndarray, original_probs: np.ndarray, original_fdc: np.ndarray):
    """
    Given an FDC original_fdc for the exceedences given in original_probs, resample it
    to new_probs. This uses a spline and will extrapolate if outside the domain.

    Parameters
    ----------
    new_probs: np.ndarray
        exceedance probabilities for the resampled FDC
    original_probs: np.ndarray
        original exceedance probabilities of the FDC
    original_fdc: np.ndarray
        original FDC values for the original exceedance probabilities.

    Returns
    -------

    """
    new_fdc = interp1d(original_probs, original_fdc,
                       kind='cubic', fill_value='extrapolate')(new_probs)

    # print(original_probs, new_probs)
    # plt.plot(new_probs, new_fdc, label="New")
    # plt.plot(original_probs, original_fdc, linestyle="--", label="Orig")
    # plt.legend()
    # plt.show()
    return new_fdc


def calculate_parameters_fdc(distribution, original_fdc):
    """
    Approximate a distribution fitting the hydrological time-series.

    Parameters
    ----------
    distribution
    original_fdc

    Returns
    -------

    """
    dist = getattr(scipy.stats, distribution)
    params = dist.fit(original_fdc)
    return params


def join_data_fdc(fdc_ds: xr.Dataset, characteristics_df: pd.DataFrame,
                  method: str, selected_characteristics: Sequence[str],
                  categorical_characteristics: Sequence[str] = []) -> Tuple[pd.DataFrame, Sequence[str], Sequence[str]]:
    """
    Prepare a pandas dataframe with characteristics and FDC values

    Parameters
    ----------
    fdc_ds
    characteristics_df
    method
    selected_characteristics
    categorical_characteristics

    Returns
    -------

    """
    rchids_with_characteristics = characteristics_df.index
    rchids_with_fdc = fdc_ds.station_rchid.values

    selected_characteristics_with_categorical = selected_characteristics.copy()

    # down-sampled probabilities
    probabilities_fdc = np.array([0.0001, 0.0003, 0.001, 0.005, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                  0.6, 0.7, 0.8, 0.9, 0.95, 0.9950, 0.999, 0.9997, 0.9999])

    distribution = None
    columns_result = probabilities_fdc
    fdc_type = method.split("_")[1]
    if fdc_type == "parameter":
        distribution = method.split("_")[2]
        columns_result = ["distribution", "param0", "param1", "param2"]

    # get common reaches
    common_reaches = list(sorted(set(rchids_with_characteristics).intersection(rchids_with_fdc)))
    exceedence_rates = fdc_ds.percentile.values

    # Hot encode the categorical characteristics
    for categorical in categorical_characteristics:
        # Getting some literal columns
        hot_encoding = pd.get_dummies(characteristics_df[categorical], prefix=categorical)
        hot_encoding_cols = list(hot_encoding.columns)
        characteristics_df = characteristics_df.join(hot_encoding)
        selected_characteristics_with_categorical += hot_encoding_cols

    training_df = pd.DataFrame(index=common_reaches,
                               columns=selected_characteristics_with_categorical + list(columns_result))

    for reach in common_reaches:
        characteristics_reach = characteristics_df.loc[reach][selected_characteristics_with_categorical]
        fdc_idx = np.where(fdc_ds.station_rchid.values == reach)[0][0]
        fdc_reach = fdc_ds.Obs_FDC[fdc_idx, :].values

        # rescale the values
        fdc_reach = np.log10(fdc_reach / characteristics_df.loc[reach]["uparea"])

        if fdc_type == "discrete":
            fdc = resample_fdc(probabilities_fdc, exceedence_rates, fdc_reach)
        else:
            fdc = [distribution] + list(fdc_ds.variables["Obs_FDC_{}".format(distribution)][fdc_idx, :].values)

        # insert fdc values and characteristics in the resulting df
        training_df.loc[reach, selected_characteristics_with_categorical] = characteristics_reach
        training_df.loc[reach, columns_result] = fdc

    return training_df, selected_characteristics_with_categorical, columns_result


def plot_FDC(rchid: int, x: Sequence[str], y_test, y_pred, output_dir=".", score=None):
    x = np.array(list(map(float, x)))
    plt.figure(figsize=(10, 8))
    plt.plot(x, y_test.iloc[0].values, label="original")
    plt.plot(x, y_pred[0], label="estimated")
    if score is not None:
        plt.text(0.7, -0.5, "MSE={:.3f}".format(score), fontsize=14)
    plt.legend()
    plt.xlabel("Exceedance")
    plt.ylabel("Flow log10(m3/s/km2)")
    plt.title("Reach: {}".format(rchid))
    # plt.show()
    plt.savefig("{}/FDC_estimated_{}.png".format(output_dir, rchid))
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_fdc_nc")
    parser.add_argument("properties_csv")
    parser.add_argument("method", choices=["rf_discrete", "mnn_discrete",
                                           "rf_parameter_genextreme", "rf_parameter_lognorm"])
    parser.add_argument("-output_training_data", help="Output file containing predictors and prediction for training",
                        default="training_data.csv")
    parser.add_argument("-output_dir", help="Directory that will contain the results", default=".")
    return parser.parse_args()


def main():
    args = parse_args()
    input_fdc_nc = args.input_fdc_nc
    properties_csv = args.properties_csv
    output_joint_training_csv = args.output_training_data
    output_dir = args.output_dir
    method = args.method

    input_fdc_ds = xr.open_dataset(input_fdc_nc)
    properties_df = pd.read_csv(properties_csv, index_col="rchid")
    model = method.split("_")[0]
    fdc_type = method.split("_")[1]

    try:
        distribution = method.split("_")[2]
    except IndexError:
        distribution = None

    filters = {'GoodSites': True, 'IsWeir': False}
    for k, v in filters.items():
        properties_df = properties_df.loc[properties_df[k] == v]

    # This is for NIWA data, CAMELS doesn't have the same...
    selected_characteristics = ["log10_elevation", "usRainDays10", "usPET", "usAnRainVar", "usParticleSize",
                                "usAveSlope", "log10_uparea_m2"]
    categorical_characteristics = ["CLIMATE"]

    joint_fdc_characteristics, all_characteristics_list, all_results_list = join_data_fdc(input_fdc_ds, properties_df,
                                                                                          method,
                                                                                          selected_characteristics,
                                                                                          categorical_characteristics)
    print(joint_fdc_characteristics)

    joint_fdc_characteristics.to_csv(output_joint_training_csv)

    # Prepare the data for the ML
    X, Y = joint_fdc_characteristics[all_characteristics_list], joint_fdc_characteristics[all_results_list]
    cv = sklearn.model_selection.LeaveOneOut()
    if fdc_type == "parameter":
        Y.drop(columns="distribution", inplace=True)

    estimated_fdc_csv = joint_fdc_characteristics[all_results_list].copy()

    for train_index, test_index in cv.split(X.index):
        print(test_index)

        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]
        rchid = X_test.index.values[0]
        print(rchid)
        other_arguments = {}
        if model == "rf":
            regressor = regression_models.create_random_forest_model()
        else:
            other_arguments = {'batch_size': 32, 'epochs': 50, 'verbose': 0}
            regressor = regression_models.create_singleoutputsingle_model(all_characteristics_list, all_results_list)

        regressor.fit(X_train, Y_train, **other_arguments)
        Y_pred = regressor.predict(X_test)
        score = np.sqrt(mean_squared_error(Y_test, Y_pred))
        print("MSE score: ", score)

        if fdc_type == "discrete":
            plot_FDC(rchid, all_results_list, y_test=Y_test, y_pred=Y_pred, output_dir=output_dir, score=score)
        else:
            print(Y_pred, Y_test)

        if fdc_type == "parameter":
            estimated_fdc_csv.loc[rchid, :] = [distribution] + list(Y_pred[0])
            continue

        estimated_fdc_csv.loc[rchid, :] = Y_pred[0]

    # TODO: create a netcdf file that has the same format as the FDC one, but this time it has the prediction
    if fdc_type == "parameter":
        estimated_fdc_csv["distribution"] = distribution

    estimated_fdc_csv.to_csv("estimated_fdc_{}".format(method))


if __name__ == "__main__":
    main()
