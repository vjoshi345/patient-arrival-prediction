import numpy as np
from utils import mae, mse
from pmdarima.arima import auto_arima


def run_model(sev='ESI 3', forecast_range='daily'):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/{forecast_range}_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/{forecast_range}_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)

    # Model 1: Auto ARIMA without seasonality
    print('\n *** Use auto_arima method to choose the best ARIMA model (without seasonality) ***')
    arima_model = auto_arima(x_train, start_p=2, max_p=30, start_q=2, max_q=30, d=1, seasonal=False, stepwise=False,
                             random=True, random_state=101, n_fits=50, suppress_warnings=True, trace=True)
    print(arima_model.summary())
    arima_forecast = arima_model.fit(x_train).predict(90)
    print('MSE (manual):', mse(x_valid, arima_forecast))
    print('MAE (manual):', mae(x_valid, arima_forecast))

    # Model 2: Auto ARIMA with seasonality
    print('\n *** Use auto_arima method to choose the best ARIMA model (with seasonality) ***')
    arima_model_s = auto_arima(x_train, start_p=2, max_p=30, start_q=2, max_q=30, d=1, seasonal=True, m=7,
                               stepwise=False, random=True, random_state=101, n_fits=50, suppress_warnings=True,
                               trace=True)
    print(arima_model_s.summary())
    arima_forecast_s = arima_model_s.fit(x_train).predict(90)
    print('MSE (manual):', mse(x_valid, arima_forecast_s))
    print('MAE (manual):', mae(x_valid, arima_forecast_s))


if __name__ == '__main__':
    # Hourly forecast for a day (validated over last 90 days)
    run_model(sev='Total', forecast_range='daily')
