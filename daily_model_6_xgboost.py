import numpy as np
from xgboost import XGBRegressor
from itertools import product
from utils import mae, mse, mape


def windowed_dataset(series, win_size):
    """Generates dataset windows

    Args:
      series (1-D numpy array of float) - contains the values of the time series
      win_size (int) - the number of time steps to include in the feature

    Returns:
      x_data (2-D numpy array) - numpy array of x values of shape (,window_size)
      y_data (1-D numpy array) - numpy array of y values of shape (,1)
    """
    n = series.shape[0]
    x_data, y_data = [], []
    for i in range(n-win_size):
        x_data.append(series[i:i+win_size])
        y_data.append(series[i+win_size])
    x_data, y_data = np.array(x_data), np.array(y_data)
    return x_data, y_data


def model_forecast(model, train_series, win_size, n_step):
    """Uses an input model to generate xgboost predictions
    Each new forecast is used as a data point to generate
    the prediction for the next time step

    Args:
      model (XGBRegressor Model) - xgboost trained model
      train_series (array of float) - contains the values of the time series used for training
      win_size (int) - the number of time steps to include in the window
      n_step (int) - number of time steps to forecast

    Returns:
      forecast (list) - array containing predictions
    """
    forecast = []
    data_train_list = list(train_series)
    for i in range(n_step):
        if i < win_size:
            x_valid = data_train_list[-(win_size - i):] + forecast[:i]
        else:
            x_valid = forecast[-win_size:]
        x_valid = np.array(x_valid).reshape((-1, win_size))
        forecast = forecast + model.predict(x_valid).tolist()
    return forecast


def run_model(sev='Total', window_size=7):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    data_train = np.load(f'training_data/daily_{sev}_train.npy', allow_pickle=True)
    data_valid = np.load(f'training_data/daily_{sev}_valid.npy', allow_pickle=True)
    print(data_train.shape)
    print(data_valid.shape)

    # Generate the dataset windows
    x_train, y_train = windowed_dataset(data_train, window_size)

    # Train xgboost regression model
    print(f'\n *** Metrics for xgboost linear model (with window size={window_size}) ***')
    n_est_lis, lr_lis, booster_lis = [50, 100, 200], [0.001, 0.01, 0.1, 1], ['gbtree', 'gblinear']
    config = list(product(n_est_lis, lr_lis, booster_lis))
    best_mse, best_mae, best_params = float('Inf'), float('Inf'), None
    for params in config:
        print('******************************')
        print('Training for params:', params)
        model = XGBRegressor(objective='reg:squarederror', n_estimators=params[0],
                             learning_rate=params[1], booster=params[2], verbosity=1)
        model.fit(x_train, y_train)

        # Predict for the last 90 days: use each new forecast to predict the next value
        try:
            forecast = model_forecast(model, data_train, window_size, 90)
        except Exception:
            continue
        forecast = np.array(forecast)

        mse_manual = mse(data_valid, forecast)
        mae_manual = mae(data_valid, forecast)
        mape_manual = mape(data_valid, forecast)
        print('MSE (manual):', mse_manual)
        print('MAE (manual):', mae_manual)
        print('MAPE (manual):', mape_manual)

        if mse_manual < best_mse:
            best_mse, best_mae, best_mape = mse_manual, mae_manual, mape_manual
            best_params = params
    print('\nBest performance obtained with:', best_params)
    print('MSE:', best_mse)
    print('MAE:', best_mae)
    print('MAPE:', best_mape)


if __name__ == '__main__':
    run_model(sev='Total', window_size=30)
