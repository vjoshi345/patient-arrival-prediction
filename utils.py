import numpy as np


def moving_average_forecast(series, window_size):
    """Generates a moving average forecast

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to compute the average for

    Returns:
      forecast (array of float) - the moving average forecast
    """

    # Initialize a list
    forecast = []

    # Compute the moving average based on the window size
    for time in range(len(series) - window_size):
        forecast.append(series[time:time + window_size].mean())

    # Convert to a numpy array
    forecast = np.array(forecast)

    return forecast


def mse(ar1, ar2):
    return np.mean(np.square(ar1 - ar2))


def mae(ar1, ar2):
    return np.mean(np.abs(ar1 - ar2))


def mape(cnt_arr, var_arr):
    return np.mean(np.abs(np.divide(cnt_arr - var_arr, cnt_arr)))
