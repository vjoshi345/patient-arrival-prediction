import numpy as np
from statsmodels.tsa.api import Holt
from utils import mae, mse


def run_model(sev='ESI 3', forecast_range='daily'):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/{forecast_range}_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/{forecast_range}_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)

    # Model 1: Holt's linear trend
    print('\n *** Metrics for the Holt\'s linear trend method ***')
    holt_fit = Holt(x_train, initialization_method="estimated").fit()
    holt_forecast = holt_fit.forecast(90)
    print('First 10 values of the forecast:', holt_forecast[0:10])

    print('MSE (manual):', mse(x_valid, holt_forecast))
    print('MAE (manual):', mae(x_valid, holt_forecast))

    # Model 2: Holt's additive damped trend
    print('\n *** Metrics for the Holt\'s additive damped trend method ***')
    holt_add_fit = Holt(x_train, damped_trend=True, initialization_method="estimated").fit()
    holt_add_forecast = holt_add_fit.forecast(90)
    print('First 10 values of the forecast:', holt_add_forecast[0:10])

    print('MSE (manual):', mse(x_valid, holt_add_forecast))
    print('MAE (manual):', mae(x_valid, holt_add_forecast))


if __name__ == '__main__':
    # Daily patient arrival forecast (validated over last 90 days)
    run_model(sev='Total', forecast_range='daily')
