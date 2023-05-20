import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from utils import mae, mse, mape


def run_model(sev='ESI 3'):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/weekly_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/weekly_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)
    x_valid_flatten = np.ndarray.flatten(x_valid, order='F')
    print(x_valid_flatten.shape)

    # NOTE: the time series values must be strictly positive while using multiplicative trend or seasonal
    # components. Therefore, in the below we only utilize additive trend/seasonality
    # Model 1: Holt-Winters' no trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' no trend-additive seasonal ***')
    holt_add_sea_forecast = list()
    for i in range(7):
        train_list = list(x_train[i, :])
        forecast = []
        holt_add_sea_fit = ExponentialSmoothing(train_list, seasonal_periods=4, trend=None, seasonal='add',
                                                initialization_method="estimated").fit()
        forecast.append(list(holt_add_sea_fit.forecast(4)))
        holt_add_sea_forecast.append(forecast)
    holt_add_sea_forecast = np.asarray(holt_add_sea_forecast)
    holt_add_sea_forecast_flatten = np.ndarray.flatten(holt_add_sea_forecast, order='F')
    print('MSE (manual):', mse(x_valid_flatten, holt_add_sea_forecast_flatten))
    print('MAE (manual):', mae(x_valid_flatten, holt_add_sea_forecast_flatten))

    # Model 2: Holt-Winters' additive trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' additive trend-additive seasonal ***')
    holt_add_add_forecast = list()
    for i in range(7):
        train_list = list(x_train[i, :])
        forecast = []
        holt_add_add_fit = ExponentialSmoothing(train_list, seasonal_periods=4, trend='add', seasonal='add',
                                                initialization_method="estimated").fit()
        forecast.append(list(holt_add_add_fit.forecast(4)))
        holt_add_add_forecast.append(forecast)
    holt_add_add_forecast = np.asarray(holt_add_add_forecast)
    holt_add_add_forecast_flatten = np.ndarray.flatten(holt_add_add_forecast, order='F')
    print('MSE (manual):', mse(x_valid_flatten, holt_add_add_forecast_flatten))
    print('MAE (manual):', mae(x_valid_flatten, holt_add_add_forecast_flatten))
    print('MAPE (manual):', mape(x_valid_flatten, holt_add_add_forecast_flatten))

    # Model 3: Holt-Winters' additive damped trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' additive damped trend-additive seasonal ***')
    holt_add_damped_add_forecast = list()
    for i in range(7):
        train_list = list(x_train[i, :])
        forecast = []
        holt_add_damped_add_fit = ExponentialSmoothing(train_list, seasonal_periods=4, trend='add', seasonal='add',
                                                       damped_trend=True, initialization_method="estimated").fit()
        forecast.append(list(holt_add_damped_add_fit.forecast(4)))
        holt_add_damped_add_forecast.append(forecast)
    holt_add_damped_add_forecast = np.asarray(holt_add_damped_add_forecast)
    holt_add_damped_add_forecast_flatten = np.ndarray.flatten(holt_add_damped_add_forecast, order='F')
    print('MSE (manual):', mse(x_valid_flatten, holt_add_damped_add_forecast_flatten))
    print('MAE (manual):', mae(x_valid_flatten, holt_add_damped_add_forecast_flatten))


if __name__ == '__main__':
    # Daily forecast for a week (validated over last 4 weeks)
    run_model(sev='Total')
