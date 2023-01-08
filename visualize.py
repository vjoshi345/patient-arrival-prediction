import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


def parse_func(date_col):
    return [datetime.strptime(d, '%m/%d/%Y %H:%M') for d in date_col]


def plot_series(time, series, format="-", start=0, end=None, label=None):
    # Setup dimensions of the graph figure
    plt.figure(figsize=(18, 8))

    # Plot the time series data
    plt.plot(time[start:end], series[start:end], format)
    plt.ylim(-20, 50)

    # Label the axes
    plt.xlabel("Hour")
    plt.ylabel("Patient Count")

    if label:
        plt.legend(fontsize=14, labels=label)

    # Overlay a grid on the graph
    plt.grid(True)

    # Draw the graph on screen
    plt.show()


INP_FILE = 'data/patient_arrival_data.csv'

# Read the patient arrival and process the Date column
data = pd.read_csv(INP_FILE, parse_dates=['Date'], date_parser=parse_func)
data['year'] = [d.year for d in data['Date']]
data['month'] = [d.month for d in data['Date']]
data['day'] = [d.day for d in data['Date']]
data['hour'] = [d.hour for d in data['Date']]
print(data.shape)

time_step = range(len(data['ESI 3']))
plot_series(time=time_step[0:672], series=data['ESI 3'][0:672])

