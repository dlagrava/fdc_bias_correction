import regression_models

import xarray as xr
import pandas as pd
import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
                       kind='slinear', fill_value='extrapolate')(new_probs)
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
    exceedence_rates = fdc_ds.exceed.values

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

        # rescale the values (excepting the categorical)
        fdc_reach = np.log10(fdc_reach / characteristics_df.loc[reach]["uparea"])

        if fdc_type == "discrete":
            fdc = resample_fdc(probabilities_fdc, exceedence_rates, fdc_reach)
        else:
            assert False, "This is not working yet" # TODO: fix the calculation to come from the time-series
            assert distribution is not None, "Distribution should not be None"
            fdc = calculate_parameters_fdc(distribution, fdc_reach)

        # insert fdc values and characteristics in the resulting df
        training_df.loc[reach, selected_characteristics_with_categorical] = characteristics_reach
        training_df.loc[reach, columns_result] = [distribution] + list(fdc)

    return training_df, selected_characteristics_with_categorical, columns_result


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
    model = method.split("_")[0]

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

    joint_fdc_characteristics.to_csv("training_data_fdc.csv")

    # Prepare the data for the ML
    X, Y = joint_fdc_characteristics[all_characteristics_list], joint_fdc_characteristics[all_results_list]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
    print(X_test, Y_test)
    if model == "rf":
        regressor = regression_models.create_random_forest_model()
        regressor.fit(X_train, Y_train)
    else:
        regressor = regression_models.create_singleoutputsingle_model(all_characteristics_list, all_results_list)
        regressor.fit(X_train, Y_train, batch_size=32, epochs=50, verbose=0)

    Y_pred = regressor.predict(X_test)
    print(Y_pred, Y_test)
    score = np.sqrt(mean_squared_error(Y_test, Y_pred))
    print("MSE score: ", score)


if __name__ == "__main__":
    main()
