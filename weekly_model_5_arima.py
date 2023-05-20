import numpy as np
from pmdarima.arima import auto_arima
from utils import mae, mse, mape


def run_model(sev='ESI 3'):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/weekly_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/weekly_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)
    x_valid_flatten = np.ndarray.flatten(x_valid, order='F')
    print(x_valid_flatten.shape)

    # Model 1: Auto ARIMA without seasonality
    print('\n *** Use auto_arima method to choose the best ARIMA model (without seasonality) ***')
    arima_forecast = list()
    for i in range(7):
        train_list = list(x_train[i, :])
        forecast = []
        arima_model = auto_arima(train_list, start_p=2, start_q=2, d=1, seasonal=False,
                                 stepwise=True, suppress_warnings=True, trace=True)
        print(arima_model.summary())
        forecast.append(arima_model.fit(train_list).predict(4))
        arima_forecast.append(forecast)
    arima_forecast = np.asarray(arima_forecast)
    arima_forecast_flatten = np.ndarray.flatten(arima_forecast, order='F')

    # NOTE: this model takes a very long time to train. Hence, currently this is commented out
    # Model 2: Auto ARIMA with seasonality
    print('\n *** Use auto_arima method to choose the best ARIMA model (with seasonality) ***')
    arima_forecast_s = list()
    for i in range(7):
        train_list = list(x_train[i, :])
        forecast = []
        arima_model_s = auto_arima(train_list, start_p=2, start_q=2, d=1, seasonal=True, m=4,
                                   stepwise=True, suppress_warnings=True, trace=True)
        print(arima_model_s.summary())
        forecast.append(arima_model_s.fit(train_list).predict(4))
        arima_forecast_s.append(forecast)
    arima_forecast_s = np.asarray(arima_forecast_s)
    arima_forecast_s_flatten = np.ndarray.flatten(arima_forecast_s, order='F')

    print('\n ******** Printing out the metrics for both models ********')
    print('*** ARIMA without seasonality ***')
    print('MSE (manual):', mse(x_valid_flatten, arima_forecast_flatten))
    print('MAE (manual):', mae(x_valid_flatten, arima_forecast_flatten))
    print('MAPE (manual):', mape(x_valid_flatten, arima_forecast_flatten))
    print('\n *** ARIMA with seasonality ***')
    print('MSE (manual):', mse(x_valid_flatten, arima_forecast_s_flatten))
    print('MAE (manual):', mae(x_valid_flatten, arima_forecast_s_flatten))
    print('MAPE (manual):', mape(x_valid_flatten, arima_forecast_s_flatten))


if __name__ == '__main__':
    # Daily forecast for a week (validated over last 4 weeks)
    run_model(sev='Total')
