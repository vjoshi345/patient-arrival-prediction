import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from utils import mae, mse, mape


def run_model(sev='ESI 3', forecast_range='daily'):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/{forecast_range}_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/{forecast_range}_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)

    # NOTE: the time series values must be strictly positive while using multiplicative trend or seasonal
    # components. Therefore, in the below we only utilize additive trend/seasonality
    # Model 1: Holt-Winters' no trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' no trend-additive seasonal ***')
    holt_add_sea_fit = ExponentialSmoothing(x_train, seasonal_periods=7, trend=None, seasonal='add',
                                            initialization_method="estimated").fit()
    holt_add_sea_forecast = holt_add_sea_fit.forecast(90)
    print('First 10 values of the forecast:', holt_add_sea_forecast[0:10])

    print('MSE (manual):', mse(x_valid, holt_add_sea_forecast))
    print('MAE (manual):', mae(x_valid, holt_add_sea_forecast))

    # Model 2: Holt-Winters' additive trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' additive trend-additive seasonal ***')
    holt_add_add_fit = ExponentialSmoothing(x_train, seasonal_periods=7, trend='add', seasonal='add',
                                            initialization_method="estimated").fit()
    holt_add_add_forecast = holt_add_add_fit.forecast(90)
    print('First 10 values of the forecast:', holt_add_add_forecast[0:10])

    print('MSE (manual):', mse(x_valid, holt_add_add_forecast))
    print('MAE (manual):', mae(x_valid, holt_add_add_forecast))
    print('MAPE (manual):', mape(x_valid, holt_add_add_forecast))

    # Model 3: Holt-Winters' additive damped trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' additive damped trend-additive seasonal ***')
    holt_add_damped_add_fit = ExponentialSmoothing(x_train, seasonal_periods=7, trend='add', seasonal='add',
                                                   damped_trend=True, initialization_method="estimated").fit()
    holt_add_damped_add_forecast = holt_add_damped_add_fit.forecast(90)

    print('MSE (manual):', mse(x_valid, holt_add_damped_add_forecast))
    print('MAE (manual):', mae(x_valid, holt_add_damped_add_forecast))


if __name__ == '__main__':
    # Daily patient arrival forecast (validated over last 90 days)
    run_model(sev='Total', forecast_range='daily')
