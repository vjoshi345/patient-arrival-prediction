import pandas as pd
import numpy as np
import tensorflow as tf
from utils import mae, mse, moving_average_forecast

INP_FILE = 'data/patient_arrival_data.csv'

# Read the patient arrival and process the Date column
data = pd.read_csv(INP_FILE, dtype={'ESI 3': float})
print(data.shape)
print(data.dtypes)

# Split the dataset into train and validation sets
# (3.5 years for training and 0.5 years for validation)
split_time = 30696

# Get the train and validation sets
x_train = np.array(data['ESI 3'][:split_time])
x_valid = np.array(data['ESI 3'][split_time:])
print(x_train.shape)
print(x_train[0:10])
print(x_valid.shape)
print(x_valid[0:10])

# Model 1: naive forecast
print('\n *** Metrics for the naive forecast ***')
naive_forecast = np.array(data['ESI 3'][split_time - 1:-1])
print('MSE (tf):', tf.keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print('MSE (manual):', mse(x_valid, naive_forecast))
print('MAE (tf):', tf.keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())
print('MAE (manual):', mae(x_valid, naive_forecast))

# Model 2: moving average forecast
print('\n *** Metrics for the moving average forecast ***')
n_points = 168  # last 7 days
moving_avg = moving_average_forecast(np.array(data['ESI 3']), n_points)[split_time - n_points:]
print('MSE (tf):', tf.keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print('MSE (manual):', mse(x_valid, moving_avg))
print('MAE (tf):', tf.keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())
print('MAE (manual):', mae(x_valid, moving_avg))

# Model 3: moving average with differencing
print('\n *** Metrics for moving average with differencing ***')
n_avg = 168  # last 7 days
n_seasonality = 168  # last 7 days
series = np.array(data['ESI 3'])

# Compute the series with differencing and then do a moving average
diff_series = (series[n_seasonality:] - series[:-n_seasonality])
diff_moving_avg = moving_average_forecast(diff_series, n_avg)

# Slice the prediction points that corresponds to the validation set time steps
diff_moving_avg = diff_moving_avg[split_time - n_seasonality - n_avg:]

# Add the trend and seasonality from the original series
diff_moving_avg_plus_past = series[split_time - n_seasonality:-n_seasonality] + diff_moving_avg

print('MSE (tf):', tf.keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print('MSE (manual):', mse(x_valid, diff_moving_avg_plus_past))
print('MAE (tf):', tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())
print('MAE (manual):', mae(x_valid, diff_moving_avg_plus_past))

# Model 4: moving average with differencing (smoothed past)
print('\n *** Metrics for moving average with differencing (smoothed past) ***')
diff_moving_avg_plus_smooth_past = moving_average_forecast(
    series[split_time - (n_seasonality + 5):-(n_seasonality - 6)],
    11) + diff_moving_avg
print('MSE (tf):', tf.keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print('MSE (manual):', mse(x_valid, diff_moving_avg_plus_smooth_past))
print('MAE (tf):', tf.keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy())
print('MAE (manual):', mae(x_valid, diff_moving_avg_plus_smooth_past))
