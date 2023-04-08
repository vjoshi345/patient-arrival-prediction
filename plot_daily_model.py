import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing


def run_model(sev):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/daily_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/daily_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)

    # Model 1: Holt-Winters' no trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' no trend-additive seasonal ***')
    holt_add_sea_fit = ExponentialSmoothing(x_train, seasonal_periods=7, trend=None, seasonal='add',
                                            initialization_method="estimated").fit()
    holt_add_sea_forecast = holt_add_sea_fit.forecast(90)

    # Plot for Holt Winters no trend-additive seasonal
    plt.figure()
    plt.plot(x_valid, color='blue', label='actual')
    plt.plot(holt_add_sea_forecast, color='green', label='forecast')
    plt.title(f'Daily forecast: {sev}')
    plt.legend()
    plt.savefig(f'results/daily_{sev}.png')
    plt.show()


if __name__ == '__main__':
    # Daily patient arrival forecast (validated over last 90 days)
    run_model(sev='Total')
