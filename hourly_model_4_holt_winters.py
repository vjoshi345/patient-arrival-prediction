import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
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

    # NOTE: the time series values must be strictly positive while using multiplicative trend or seasonal
    # components. Therefore, in the below we only utilize additive trend/seasonality
    # Model 1: Holt-Winters' additive trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' additive trend-additive seasonal ***')
    holt_add_add_forecast = list()
    for i in range(24):
        train_list = list(x_train[i, :])
        forecast = []
        holt_add_add_fit = ExponentialSmoothing(train_list, seasonal_periods=7, trend='add', seasonal='add',
                                                initialization_method="estimated").fit()
        forecast.append(list(holt_add_add_fit.forecast(90)))
        holt_add_add_forecast.append(forecast)
    holt_add_add_forecast = np.asarray(holt_add_add_forecast)
    holt_add_add_forecast_flatten = np.ndarray.flatten(holt_add_add_forecast, order='F')
    print('MSE (manual):', mse(x_valid_flatten, holt_add_add_forecast_flatten))
    print('MAE (manual):', mae(x_valid_flatten, holt_add_add_forecast_flatten))

    # Model 2: Holt-Winters' additive damped trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' additive damped trend-additive seasonal ***')
    holt_add_damped_add_forecast = list()
    for i in range(24):
        train_list = list(x_train[i, :])
        forecast = []
        holt_add_damped_add_fit = ExponentialSmoothing(train_list, seasonal_periods=7, trend='add', seasonal='add',
                                                       damped_trend=True, initialization_method="estimated").fit()
        forecast.append(list(holt_add_damped_add_fit.forecast(90)))
        holt_add_damped_add_forecast.append(forecast)
    holt_add_damped_add_forecast = np.asarray(holt_add_damped_add_forecast)
    holt_add_damped_add_forecast_flatten = np.ndarray.flatten(holt_add_damped_add_forecast, order='F')
    print('MSE (manual):', mse(x_valid_flatten, holt_add_damped_add_forecast_flatten))
    print('MAE (manual):', mae(x_valid_flatten, holt_add_damped_add_forecast_flatten))


if __name__ == '__main__':
    # Hourly forecast for a day (validated over last 90 days)
    run_model(sev='ESI 1', forecast_range='hourly')
