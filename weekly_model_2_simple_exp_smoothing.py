import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
from utils import mae, mse


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

    # Parameters of the model

    # Model 1: simple exponential smoothing
    print('\n *** Metrics for the Simple Exponential Smoothing ***')
    simp_exp_forecast = list()
    for i in range(7):
        train_list = list(x_train[i, :])
        valid_list = list(x_valid[i, :])
        forecast = []
        for j in range(4):
            simp_exp_fit = SimpleExpSmoothing(train_list + valid_list[0:j], initialization_method="estimated").fit()
            forecast.append(list(simp_exp_fit.forecast(1)))
        simp_exp_forecast.append(forecast)
    simp_exp_forecast = np.asarray(simp_exp_forecast)
    simp_exp_forecast_flatten = np.ndarray.flatten(simp_exp_forecast, order='F')

    print('MSE (manual):', mse(x_valid_flatten, simp_exp_forecast_flatten))
    print('MAE (manual):', mae(x_valid_flatten, simp_exp_forecast_flatten))


if __name__ == '__main__':
    # Daily forecast for a week (validated over last 4 weeks)
    run_model(sev='Total')
