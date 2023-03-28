import numpy as np
from xgboost import XGBRegressor
from itertools import product
from utils import mae, mse


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


def run_model(sev='ESI 3', window_size=7):
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
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, booster='gbtree', verbosity=1)
    model.fit(x_train, y_train)

    # Predict for the last 90 days: use each new forecast to predict the next value
    forecast = model_forecast(model, data_train, window_size, 90)
    forecast = np.array(forecast)

    print(f'\n *** Metrics for xgboost linear model (with window size={window_size}) ***')
    mse_manual = mse(data_valid, forecast)
    print('MSE (manual):', mse_manual)
    mae_manual = mae(data_valid, forecast)
    print('MAE (manual):', mae_manual)

    return mse_manual, mae_manual


if __name__ == '__main__':
    run_model(sev='Total', window_size=7)
    # sev_list = ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5', 'Total']
    # out_dict = {}
    # for pat_sev in sev_list:
    #     mse_man, mae_man = run_model(sev=pat_sev)
    #     out_dict[pat_sev] = [mse_man, mae_man]
    # out_df = pd.DataFrame.from_dict(out_dict, orient='index', columns=['MSE', 'MAE'])
    # print(out_df.shape)
    # print(out_df.head(n=6))
    # out_df.to_csv('results/daily_pred_xgboost.csv')
