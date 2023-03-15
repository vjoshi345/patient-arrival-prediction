import numpy as np
from statsmodels.tsa.api import SimpleExpSmoothing
from utils import mae, mse


def run_model(sev='ESI 3', forecast_range='daily'):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/{forecast_range}_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/{forecast_range}_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)

    # Model 1: simple exponential smoothing
    print('\n *** Metrics for the Simple Exponential Smoothing ***')
    simp_exp_fit = SimpleExpSmoothing(x_train, initialization_method="estimated").fit()
    forecast = simp_exp_fit.forecast(90)
    print('First 10 values of the forecast:', forecast[0:10])

    print('MSE (manual):', mse(x_valid, forecast))
    print('MAE (manual):', mae(x_valid, forecast))


if __name__ == '__main__':
    # Daily patient arrival forecast (validated over last 90 days)
    run_model(sev='Total', forecast_range='daily')
