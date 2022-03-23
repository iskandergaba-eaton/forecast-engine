# Import librairies
from engine_deep import EatonPowerForecastModel
from util import ungap
from util import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = 18, 8

np.random.seed(42)

if __name__ == "__main__":
    # Global variables
    k, h, lag, threshold = (10, 1, 10, 1)
    influence = 1
    # Data loading
    DCs = ["simtcsvc01", "usltccnvcenter4",
           "usltccnvcenter5", "usltcsvcenter2", "usltcsvcenter3"]
    versions_gap = ["19-02-2022"]
    root_gap, dc_gap = '../data/clean/{}'.format(versions_gap[-1]), DCs[0]
    dfs_mean_gap, powers_mean_noisy = {}, {}
    for dc in DCs:
        dfs_mean_gap[dc] = load_data(root_gap, dc, agg_func=np.mean, freq='6H')
        print(dc, " loaded")

    for dc in DCs:
        powers_mean_noisy[dc] = ungap(dfs_mean_gap[dc], col_name='power')
        print('Ungapped {}'.format(dc))

    # Test on mean power consumption
    dc = DCs[1]
    time_window = 5
    autoencoder_activation = 'tanh'
    model_activation = 'tanh'
    n_layers_autoencoder = 200
    n_units = 5
    ts_power = powers_mean_noisy[dc]
    y = pd.Series(list(ts_power), index=range(len(ts_power)))
    # split train/test (80% ; 20%)
    split = 8*len(y)//10
    y_train, y_val = y[:split], y

    alg = EatonPowerForecastModel(
        time_window, autoencoder_activation, model_activation)
    Xtrain, ytrain = alg.initialize_dataset(y_train)
    Xval, yval = alg.initialize_dataset(y_val)

    # Scale down the data
    xtrain_std, ytrain_std, sc_xtrain, sc_ytrain = alg.prepare_data_autoencoder(
        Xtrain, ytrain)
    xval_std, yval_std, sc_xval, sc_yval = alg.prepare_data_autoencoder(
        Xval, yval)

    X_train, Y_train = alg.create_variable_for_model(
        xtrain_std, time_window), ytrain_std[time_window:]
    X_val, Y_val = alg.create_variable_for_model(
        xval_std, time_window), yval_std[time_window:]

    # Dimensionality reduction
    encoder = alg.generate_encoder(X_train, n_layers_autoencoder, time_window)
    xx_train = encoder.predict(X_train)
    xx_val = encoder.predict(X_val)

    x_train = alg.create_variable_for_model(xx_train, time_window)
    y_train = Y_train[time_window:]
    x_val = alg.create_variable_for_model(xx_val, time_window)
    y_val = Y_val[time_window:]

    # forecast
    model = alg.forecast_model(x_train, n_units, time_window)
    history = model.fit(x_train, y_train)
    # make predictions on the whole dataset (start from the beginning, for indexing issues !)
    preds = model.predict(x_val)

    # Plotting
    train_id, val_id = range(len(y_train)), range(len(y_train), len(y_val))
    plt.plot(train_id, y_val[train_id], alpha=0.5,
             color='green', label='Observed past')
    plt.plot(val_id, y_val[val_id], alpha=0.5,
             color='red', label='observed future')
    plt.plot(val_id, preds[val_id], alpha=0.5, label='Forecast')
    plt.legend(loc='upper left')
    plt.title("Forecasting using LSTM model (encoder-decoder architecture) on " + DCs[0]
              + "(sliding window =" + str(time_window) + " ; n_layers=" + str(n_layers_autoencoder) + ")")
    plt.legend()
    plt.savefig("plot.png", bbox_inches='tight', pad_inches=0.5, transparent=True)
    plt.close()
