import pandas as pd
import numpy as np
import tensorflow as tf
from itertools import product
from utils import mae, mse


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    """Generates dataset windows

    Args:
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the feature
      batch_size (int) - the batch size
      shuffle_buffer(int) - buffer size to use for the shuffle method

    Returns:
      dataset (TF Dataset) - TF Dataset containing time windows
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    # Create tuples with features and labels
    dataset = dataset.map(lambda window: (window[:-1], window[-1]))

    # Shuffle the windows
    dataset = dataset.shuffle(shuffle_buffer)

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    return dataset


def model_forecast(model, series, window_size, batch_size):
    """Uses an input model to generate predictions on data windows

    Args:
      model (TF Keras Model) - model that accepts data windows
      series (array of float) - contains the values of the time series
      window_size (int) - the number of time steps to include in the window
      batch_size (int) - the batch size

    Returns:
      forecast (numpy array) - array containing predictions
    """

    # Generate a TF Dataset from the series values
    dataset = tf.data.Dataset.from_tensor_slices(series)

    # Window the data but only take those with the specified size
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)

    # Flatten the windows by putting its elements in a single batch
    dataset = dataset.flat_map(lambda w: w.batch(window_size))

    # Create batches of windows
    dataset = dataset.batch(batch_size).prefetch(1)

    # Get predictions on the entire dataset
    forecast = model.predict(dataset)

    return forecast


INP_FILE = 'data/patient_arrival_data.csv'

# Read the patient arrival and process the Date column
data = pd.read_csv(INP_FILE, dtype={'ESI 3': float})
print(data.shape)
print(data.dtypes)

# Split the dataset into train and validation sets
# (3.5 years for training and 0.5 years for validation)
split_time = 30696

# Get the train and validation sets
series = np.array(data['ESI 3'])
x_train = np.array(data['ESI 3'][:split_time])
x_valid = np.array(data['ESI 3'][split_time:])
print(x_train.shape)
print(x_train[0:10])
print(x_valid.shape)
print(x_valid[0:10])

# RNN model parameters
window_size = 24
batch_size = 32
shuffle_buffer_size = 1000

# Generate the dataset windows
train_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

# Build the model
model = tf.keras.models.Sequential([
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[window_size]),
  tf.keras.layers.SimpleRNN(units=40),
  tf.keras.layers.Dense(units=1, activation='relu')
])
model.summary()

# Set the learning rate
learning_rate = 1e-6

# Set the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

# Set the training parameters
model.compile(loss='mse', optimizer=optimizer)

# Train the model
history = model.fit(train_dataset, epochs=50)

# Prediction
forecast_series = series[split_time - window_size:-1]
forecast = model_forecast(model, forecast_series, window_size, batch_size)
# Drop single dimensional axis
forecast = forecast.squeeze()

print(f'\n *** Metrics for RNN (with window size={window_size}) ***')
print('MSE (tf):', tf.keras.metrics.mean_squared_error(x_valid, forecast).numpy())
print('MSE (manual):', mse(x_valid, forecast))
print('MAE (tf):', tf.keras.metrics.mean_absolute_error(x_valid, forecast).numpy())
print('MAE (manual):', mae(x_valid, forecast))
