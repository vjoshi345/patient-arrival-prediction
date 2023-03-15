import numpy as np
from utils import mae, mse, moving_average_forecast


def run_model(sev='ESI 3', forecast_range='daily'):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/{forecast_range}_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/{forecast_range}_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)

    # Parameters of the model
    split_time = 1371  # this corresponds to using last 90 days for validation
    n_avg = 30  # take avg of last 30 days
    n_seasonality = 30  # 30 day seasonality for differencing
    win_size = 7  # window size for smoothening

    # Model 1: naive forecast
    print('\n *** Metrics for the naive forecast ***')
    naive_forecast = np.append(x_train[-1], x_valid[:-1])
    print(naive_forecast.shape)

    print('MSE (manual):', mse(x_valid, naive_forecast))
    print('MAE (manual):', mae(x_valid, naive_forecast))

    # Model 2: moving average forecast
    print('\n *** Metrics for the moving average forecast ***')
    moving_avg = moving_average_forecast(np.append(x_train, x_valid), n_avg)[split_time - n_avg:]
    print(moving_avg.shape)

    print('MSE (manual):', mse(x_valid, moving_avg))
    print('MAE (manual):', mae(x_valid, moving_avg))

    # Model 3 and 4: moving average with differencing
    print('\n *** Metrics for moving average with differencing ***')
    series = np.append(x_train, x_valid)

    # Compute the series with differencing and then do a moving average
    diff_series = (series[n_seasonality:] - series[:-n_seasonality])
    diff_moving_avg = moving_average_forecast(diff_series, n_avg)

    # Slice the prediction points that corresponds to the validation set time steps
    diff_moving_avg = diff_moving_avg[split_time - n_seasonality - n_avg:]

    # Add the trend and seasonality from the original series
    diff_moving_avg_plus_past = series[split_time - n_seasonality:-n_seasonality] + diff_moving_avg
    print(diff_moving_avg_plus_past.shape)

    print('MSE (manual):', mse(x_valid, diff_moving_avg_plus_past))
    print('MAE (manual):', mae(x_valid, diff_moving_avg_plus_past))

    # Model 4: moving average with differencing (smoothed past)
    print('\n *** Metrics for moving average with differencing (smoothed past) ***')
    # Add smoothed past to diff_moving_avg
    smoothed_past = moving_average_forecast(series[split_time - n_seasonality - win_size:-n_seasonality], win_size)
    diff_moving_avg_plus_smooth_past = smoothed_past + diff_moving_avg
    print(diff_moving_avg_plus_smooth_past.shape)

    print('MSE (manual):', mse(x_valid, diff_moving_avg_plus_smooth_past))
    print('MAE (manual):', mae(x_valid, diff_moving_avg_plus_smooth_past))


if __name__ == '__main__':
    # Daily patient arrival forecast (validated over last 90 days)
    run_model(sev='Total', forecast_range='daily')
