import numpy as np
import tensorflow as tf
from utils import mae, mse, mape, moving_average_forecast


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

    # Parameters of the model
    split_time = 1371  # this corresponds to using last 90 days for validation
    n_avg = 30  # take avg of last 30 days
    n_seasonality = 30  # 30 day seasonality for differencing
    win_size = 7  # window size for smoothening

    # Model 1: naive forecast
    print('\n *** Metrics for the naive forecast ***')
    naive_forecast = list()
    for i in range(24):
        naive_forecast.append(np.append(x_train[i, -1], x_valid[i, :-1]))
    naive_forecast = np.asarray(naive_forecast)
    naive_forecast_flatten = np.ndarray.flatten(naive_forecast, order='F')
    print(naive_forecast_flatten.shape)

    print('MSE (tf):', tf.keras.metrics.mean_squared_error(x_valid_flatten, naive_forecast_flatten).numpy())
    print('MSE (manual):', mse(x_valid_flatten, naive_forecast_flatten))
    print('MAE (tf):', tf.keras.metrics.mean_absolute_error(x_valid_flatten, naive_forecast_flatten).numpy())
    print('MAE (manual):', mae(x_valid_flatten, naive_forecast_flatten))
    print('MAPE (tf):', tf.keras.metrics.mean_absolute_percentage_error(x_valid_flatten, naive_forecast_flatten).numpy())
    print('MAPE (manual):', mape(x_valid_flatten, naive_forecast_flatten))

    # Model 2: moving average forecast
    print('\n *** Metrics for the moving average forecast ***')
    moving_avg = list()
    for i in range(24):
        moving_avg.append(moving_average_forecast(np.append(x_train[i, :], x_valid[i, :]), n_avg)[split_time - n_avg:])
    moving_avg = np.asarray(moving_avg)
    moving_avg_flatten = np.ndarray.flatten(moving_avg, order='F')
    print(moving_avg_flatten.shape)

    print('MSE (tf):', tf.keras.metrics.mean_squared_error(x_valid_flatten, moving_avg_flatten).numpy())
    print('MSE (manual):', mse(x_valid_flatten, moving_avg_flatten))
    print('MAE (tf):', tf.keras.metrics.mean_absolute_error(x_valid_flatten, moving_avg_flatten).numpy())
    print('MAE (manual):', mae(x_valid_flatten, moving_avg_flatten))

    # Model 3 and 4: moving average with differencing
    print('\n *** Metrics for moving average with differencing ***')
    diff_moving_avg_list = list()
    diff_moving_avg_plus_smoothed_past_list = list()

    for i in range(24):
        series = np.append(x_train[i, :], x_valid[i, :])

        # Compute the series with differencing and then do a moving average
        diff_series = (series[n_seasonality:] - series[:-n_seasonality])
        diff_moving_avg = moving_average_forecast(diff_series, n_avg)

        # Slice the prediction points that corresponds to the validation set time steps
        diff_moving_avg = diff_moving_avg[split_time - n_seasonality - n_avg:]

        # Add the trend and seasonality from the original series
        diff_moving_avg_plus_past = series[split_time - n_seasonality:-n_seasonality] + diff_moving_avg

        # Forecast with smoothed past
        smoothed_past = moving_average_forecast(
            series[split_time - n_seasonality - win_size:-n_seasonality],
            win_size)
        diff_moving_avg_plus_smooth_past = smoothed_past + diff_moving_avg

        # Append the result to the output lists
        diff_moving_avg_list.append(diff_moving_avg_plus_past)
        diff_moving_avg_plus_smoothed_past_list.append(diff_moving_avg_plus_smooth_past)

    diff_moving_avg_list = np.asarray(diff_moving_avg_list)
    diff_moving_avg_flatten = np.ndarray.flatten(diff_moving_avg_list, order='F')
    print(diff_moving_avg_flatten.shape)

    print('MSE (tf):', tf.keras.metrics.mean_squared_error(x_valid_flatten, diff_moving_avg_flatten).numpy())
    print('MSE (manual):', mse(x_valid_flatten, diff_moving_avg_flatten))
    print('MAE (tf):', tf.keras.metrics.mean_absolute_error(x_valid_flatten, diff_moving_avg_flatten).numpy())
    print('MAE (manual):', mae(x_valid_flatten, diff_moving_avg_flatten))

    # Model 4: moving average with differencing (smoothed past)
    print('\n *** Metrics for moving average with differencing (smoothed past) ***')
    diff_moving_avg_plus_smoothed_past_list = np.asarray(diff_moving_avg_plus_smoothed_past_list)
    diff_moving_avg_plus_smoothed_past_flatten = np.ndarray.flatten(diff_moving_avg_plus_smoothed_past_list, order='F')
    print('MSE (tf):', tf.keras.metrics.mean_squared_error(x_valid_flatten, diff_moving_avg_plus_smoothed_past_flatten).numpy())
    print('MSE (manual):', mse(x_valid_flatten, diff_moving_avg_plus_smoothed_past_flatten))
    print('MAE (tf):', tf.keras.metrics.mean_absolute_error(x_valid_flatten, diff_moving_avg_plus_smoothed_past_flatten).numpy())
    print('MAE (manual):', mae(x_valid_flatten, diff_moving_avg_plus_smoothed_past_flatten))


if __name__ == '__main__':
    # Hourly forecast for a day (validated over last 90 days)
    run_model(sev='Total', forecast_range='hourly')
