import numpy as np
from statsmodels.tsa.api import Holt
from utils import mae, mse


def run_model(sev='ESI 3', forecast_range='hourly'):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/{forecast_range}_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/{forecast_range}_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)
    x_valid_flatten = np.ndarray.flatten(x_valid, order='F')
    print(x_valid_flatten.shape)

    # Model 1: Holt's linear trend
    print('\n *** Metrics for the Holt\'s linear trend method ***')
    holt_forecast = list()
    for i in range(24):
        train_list = list(x_train[i, :])
        forecast = []
        holt_fit = Holt(train_list, initialization_method="estimated").fit()
        forecast.append(list(holt_fit.forecast(90)))
        holt_forecast.append(forecast)
    holt_forecast = np.asarray(holt_forecast)
    holt_forecast_flatten = np.ndarray.flatten(holt_forecast, order='F')
    print('MSE (manual):', mse(x_valid_flatten, holt_forecast_flatten))
    print('MAE (manual):', mae(x_valid_flatten, holt_forecast_flatten))

    # Model 2: Holt's additive damped trend
    print('\n *** Metrics for the Holt\'s additive damped trend method ***')
    holt_add_forecast = list()
    for i in range(24):
        train_list = list(x_train[i, :])
        forecast = []
        holt_add_fit = Holt(train_list, damped_trend=True, initialization_method="estimated").fit()
        forecast.append(list(holt_add_fit.forecast(90)))
        holt_add_forecast.append(forecast)
    holt_add_forecast = np.asarray(holt_add_forecast)
    holt_add_forecast_flatten = np.ndarray.flatten(holt_add_forecast, order='F')
    print('MSE (manual):', mse(x_valid_flatten, holt_add_forecast_flatten))
    print('MAE (manual):', mae(x_valid_flatten, holt_add_forecast_flatten))


if __name__ == '__main__':
    # Hourly forecast for a day (validated over last 90 days)
    run_model(sev='ESI 3', forecast_range='hourly')
