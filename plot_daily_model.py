import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import ExponentialSmoothing
from utils import mae, mse


def run_model(sev):
    print('\n*****************************************************')
    print('Patient Severity: ', sev)

    # Get the train and validation sets
    x_train = np.load(f'training_data/daily_{sev}_train.npy', allow_pickle=True)
    x_valid = np.load(f'training_data/daily_{sev}_valid.npy', allow_pickle=True)
    print(x_train.shape)
    print(x_valid.shape)

    # Model 1: Holt-Winters' additive trend-additive seasonal
    print('\n *** Metrics for the Holt-Winters\' additive trend-additive seasonal ***')
    holt_add_add_fit = ExponentialSmoothing(x_train, seasonal_periods=7, trend='add', seasonal='add',
                                            initialization_method="estimated").fit()
    holt_add_add_forecast = holt_add_add_fit.forecast(90)

    print('MSE (manual):', mse(x_valid, holt_add_add_forecast))
    print('MAE (manual):', mae(x_valid, holt_add_add_forecast))

    # Plot for Holt Winters additive trend-additive seasonal
    plt.figure()
    plt.plot(x_valid, color='blue', label='actual')
    plt.plot(holt_add_add_forecast, color='green', label='forecast')
    plt.title(f'Daily forecast: {sev}')
    plt.legend()
    plt.savefig(f'results/daily_{sev}.png')
    plt.show()

    # Save the actual and forecasted values
    np.save(f'results/daily_actual_{sev}.npy', arr=x_valid)
    np.save(f'results/daily_forecast_HW_add_trend_add_seasonality_{sev}.npy', arr=holt_add_add_forecast)


if __name__ == '__main__':
    # Daily patient arrival forecast (validated over last 90 days)
    run_model(sev='Total')
