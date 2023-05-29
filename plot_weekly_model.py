import numpy as np
from xgboost import XGBRegressor
from itertools import product
from utils import mae, mse, mape
import matplotlib.pyplot as plt


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


def run_model(sev='Total', window_size=4):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/weekly_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/weekly_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)
    x_valid_flatten = np.ndarray.flatten(x_valid, order='F')
    print(x_valid_flatten.shape)

    # XGBoost Regression Model
    # Predict for the last 4 weeks: use each new forecast to predict the next value
    print('\n *** Metrics for the XGBoost Regression model ***')
    n_est_lis, lr_lis, booster_lis = [50, 100, 200], [0.001, 0.01, 0.1, 1], ['gbtree', 'gblinear']
    config = list(product(n_est_lis, lr_lis, booster_lis))
    best_mse, best_mae, best_mape, best_params, best_forecast = float('Inf'), float('Inf'), float('Inf'), None, None
    for params in config:
        print('******************************')
        print('Training for params:', params)
        xgboost_forecast = list()
        for i in range(7):
            train_dow = x_train[i, :]
            x_train_dow, y_train_dow = windowed_dataset(train_dow, window_size)
            model = XGBRegressor(objective='reg:squarederror', n_estimators=params[0],
                                 learning_rate=params[1], booster=params[2], verbosity=1)
            model.fit(x_train_dow, y_train_dow)
            forecast = model_forecast(model, train_dow, window_size, 4)
            xgboost_forecast.append(forecast)
        xgboost_forecast = np.asarray(xgboost_forecast)
        xgboost_forecast_flatten = np.ndarray.flatten(xgboost_forecast, order='F')
        mse_manual = mse(x_valid_flatten, xgboost_forecast_flatten)
        mae_manual = mae(x_valid_flatten, xgboost_forecast_flatten)
        mape_manual = mape(x_valid_flatten, xgboost_forecast_flatten)
        print('Current metrics -> MSE: ', mse_manual, ' MAE: ', mae_manual, ' MAPE: ', mape_manual)
        if mse_manual < best_mse:
            best_mse, best_mae, best_mape = mse_manual, mae_manual, mape_manual
            best_forecast = xgboost_forecast_flatten
            best_params = params
    print('\nBest performance obtained with:', best_params)
    print('MSE:', best_mse)
    print('MAE:', best_mae)
    print('MAPE:', best_mape)

    # Plot for the best XGBoost model
    plt.figure()
    plt.plot(x_valid_flatten, color='blue', label='actual')
    plt.plot(best_forecast, color='green', label='forecast')
    plt.title(f'Weekly forecast (last 4 weeks): {sev}')
    plt.legend()
    plt.savefig(f'results/weekly_{sev}.png')
    plt.show()

    # Save the actual and forecasted values
    np.save(f'results/weekly_actual_{sev}.npy', arr=x_valid_flatten)
    np.save(f'results/weekly_forecast_XGBoost_{sev}.npy', arr=best_forecast)


if __name__ == '__main__':
    # Daily forecast for a week (validated over last 4 weeks)
    run_model(sev='Total', window_size=4)
