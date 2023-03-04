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


def run_model(inp_file, sev='ESI 3'):
    print('\n*****************************************************')
    print('Running for input file: ', inp_file.split('/')[-1])
    print('Patient Severity: ', sev)
    # Read the patient arrival and process the Date column
    data = pd.read_csv(inp_file, dtype={sev: float})
    print(data.shape)
    print(data.dtypes)

    # Split the dataset into train and validation sets
    # (3.5 years for training and 0.5 years for validation)
    split_time = 182

    # Get the train and validation sets
    series = np.array(data[sev])
    x_train = np.array(data[sev][:split_time])
    x_valid = np.array(data[sev][split_time:])
    print(x_train.shape)
    print(x_train[0:10])
    print(x_valid.shape)
    print(x_valid[0:10])

    # Neural Network model parameters
    window_size = 8
    act_fun = 'relu'
    batch_size = 1
    shuffle_buffer_size = 1000

    # Generate the dataset windows
    train_dataset = windowed_dataset(x_train, window_size, batch_size, shuffle_buffer_size)

    # Build the single layer neural network
    l0 = tf.keras.layers.Dense(units=1,
                               activation=act_fun,
                               input_shape=[window_size],
                               kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=43),
                               bias_initializer='zeros'
                               )
    model = tf.keras.models.Sequential([l0])
    print("Layer weights: \n {} \n".format(l0.get_weights()))
    model.summary()

    # Set the training parameters
    model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9))

    # Train the model
    model.fit(train_dataset, epochs=10)

    # Print the layer weights (after training)
    print("Layer weights after training: {}".format(l0.get_weights()))

    # Prediction
    forecast_series = series[split_time - window_size:-1]
    forecast = model_forecast(model, forecast_series, window_size, batch_size)
    # Drop single dimensional axis
    forecast = forecast.squeeze()

    print(f'\n *** Metrics for 1-layer NN (with window size={window_size} and activation function={act_fun}) ***')
    print('MSE (tf):', tf.keras.metrics.mean_squared_error(x_valid, forecast).numpy())
    mse_manual = mse(x_valid, forecast)
    print('MSE (manual):', mse_manual)
    print('MAE (tf):', tf.keras.metrics.mean_absolute_error(x_valid, forecast).numpy())
    mae_manual = mae(x_valid, forecast)
    print('MAE (manual):', mae_manual)

    return mse_manual, mae_manual


if __name__ == '__main__':
    # sev_list = ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5', 'Total']
    sev_list = ['ESI 3']
    out_dict = {}
    for pat_sev in sev_list:
        mse_man, mae_man = run_model(inp_file='data/patient_arrival_weekly.csv', sev=pat_sev)
        out_dict[pat_sev] = [mse_man, mae_man]
    out_df = pd.DataFrame.from_dict(out_dict, orient='index', columns=['MSE', 'MAE'])
    print(out_df.shape)
    print(out_df.head(n=6))
    # out_df.to_csv('data/weekly_pred_DNN.csv')
