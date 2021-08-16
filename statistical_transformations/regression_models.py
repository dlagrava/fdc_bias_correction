import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.ensemble
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.multioutput
from tensorflow import keras
from tensorflow.keras import constraints
from tensorflow.keras import layers


def create_multioutput_model(input_cols, output_cols):
    """
    This creates a MNN model as per Worland et al. 2019. It uses non negative weights to
    force the monotonic nature of FDC.

    Parameters
    ----------
    input_cols
    output_cols

    Returns
    -------

    """
    # Defining the model
    characteristics_input = keras.Input(shape=(len(input_cols)), name="characteristics")
    x = layers.Dense(35, activation="relu")(characteristics_input)
    x = layers.Dropout(0.22)(x)
    x = layers.Dense(31, activation="softmax", kernel_constraint=constraints.NonNeg())(x)
    x = layers.Dropout(0.32)(x)

    fdc_outputs = []
    for i, fdc_exceedance_value in enumerate(output_cols):
        fdc_outputs.append(keras.layers.Dense(1, activation="linear", kernel_constraint=constraints.NonNeg(),
                                              name=fdc_exceedance_value)(x))

    # Create the model
    model = keras.Model(
        inputs=[characteristics_input],
        outputs=fdc_outputs
    )

    # Compiling the model
    model.compile(
        loss=keras.losses.MeanAbsoluteError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.005)
    )

    return model


def create_singleoutputsingle_model(input_cols, output_cols):
    """
    Similar to the MNN but it outputs a single vector for the FDC.

    Parameters
    ----------
    input_cols
    output_cols

    Returns
    -------

    """
    # Defining the model
    characteristics_input = keras.Input(shape=(len(input_cols)), name="characteristics")
    x = layers.Dense(35, activation="relu")(characteristics_input)
    x = layers.Dropout(0.22)(x)
    x = layers.Dense(31, activation="softmax", kernel_constraint=constraints.NonNeg())(x)
    x = layers.Dropout(0.32)(x)

    fdc_output = layers.Dense(len(output_cols), activation="linear", kernel_constraint=constraints.NonNeg())(x)

    # Declaring model
    model = keras.Model(
        inputs=[characteristics_input],
        outputs=fdc_output
    )

    # Compiling model and testing
    model.compile(
        loss=keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(learning_rate=0.005)
    )

    return model


def create_random_forest_model(n_estimators=500, max_depth=5, criterion='mae'):
    """
    Random forest model

    Parameters
    ----------
    n_estimators
    max_depth
    criterion

    Returns
    -------

    """
    return sklearn.ensemble.RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
                                                  n_jobs=8)


def create_XGBoost_model():
    """
    This creates an XGBoost model

    Returns
    -------

    """
    model = sklearn.ensemble.GradientBoostingRegressor(n_estimators=300, learning_rate=0.05)
    return sklearn.multioutput.RegressorChain(model)
