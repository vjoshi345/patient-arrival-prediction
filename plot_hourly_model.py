import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from utils import mae, mse


def run_model(sev, plot_hour):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/hourly_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/hourly_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)
    x_valid_flatten = np.ndarray.flatten(x_valid, order='F')
    print(x_valid_flatten.shape)

    # Model 1: Holt-Winters' no trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' no trend-additive seasonal ***')
    holt_add_sea_forecast = list()
    for i in range(24):
        train_list = list(x_train[i, :])
        forecast = []
        holt_add_sea_fit = ExponentialSmoothing(train_list, seasonal_periods=7, trend=None, seasonal='add',
                                                initialization_method="estimated").fit()
        forecast.append(list(holt_add_sea_fit.forecast(90)))
        holt_add_sea_forecast.append(forecast)
    holt_add_sea_forecast = np.asarray(holt_add_sea_forecast).reshape((24, 90))
    holt_add_sea_forecast_flatten = np.ndarray.flatten(holt_add_sea_forecast, order='F')
    print('MSE (manual):', mse(x_valid_flatten, holt_add_sea_forecast_flatten))
    print('MAE (manual):', mae(x_valid_flatten, holt_add_sea_forecast_flatten))

    # Plot for Holt Winters no trend-additive seasonal
    plt.figure()
    plt.plot(x_valid[plot_hour, :], color='blue', label='actual')
    plt.plot(holt_add_sea_forecast[plot_hour, :], color='green', label='forecast')
    plt.title(f'Hourly forecast: {sev}')
    plt.legend()
    plt.savefig(f'results/hourly_{sev}.png')
    plt.show()


if __name__ == '__main__':
    # Hourly forecast for a day (validated over last 90 days)
    run_model(sev='Total', plot_hour=10)
