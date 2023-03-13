import pandas as pd
import numpy as np
from datetime import datetime

INP_FILE = 'data/patient_arrival_data.csv'


def parse_func(date_col):
    return [datetime.strptime(d, '%m/%d/%Y %H:%M') for d in date_col]


# Read the patient arrival and process the Date column
print('Running for input file: ', INP_FILE.split('/')[-1])
data = pd.read_csv(INP_FILE, parse_dates=['Date'], date_parser=parse_func,
                   dtype={'ESI 1': 'float', 'ESI 2': 'float', 'ESI 3': 'float', 'ESI 4': 'float', 'ESI 5': 'float',
                          'Total': 'float'})
data['year'] = [d.year for d in data['Date']]
data['month'] = [d.month for d in data['Date']]
data['day'] = [d.day for d in data['Date']]
data['hour'] = [d.hour for d in data['Date']]

print(data.shape)
print(data.dtypes)
print(data.head())

print('\nGroup data at a daily level')
grouped_data = data.groupby(['year', 'month', 'day'], as_index=False).agg({'ESI 1': 'sum', 'ESI 2': 'sum',
                                                                           'ESI 3': 'sum', 'ESI 4': 'sum',
                                                                           'ESI 5': 'sum', 'Total': 'sum'})
print(grouped_data.shape)
print(grouped_data.dtypes)
print(grouped_data.head())

# Confirm that sorting is not needed explicitly
print('\nSort and test that the values make sense')
sorted_grouped_data = grouped_data.sort_values(by=['year', 'month', 'day'], axis='index', ascending=True, inplace=False)
print('Comparing original data with data sorted by split columns to see if they are the same: ',
      sorted_grouped_data.equals(grouped_data))
print('No. of rows where the sum of ESIs does not equal Total:',
      sum([tot != (a + b + c + d + e) for a, b, c, d, e, tot in
           zip(grouped_data['ESI 1'], grouped_data['ESI 2'], grouped_data['ESI 3'], grouped_data['ESI 4'],
               grouped_data['ESI 5'], grouped_data['Total'])]))

for sev in ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5', 'Total']:
    print('\n**********************************')
    print(f'Patient severity: {sev}')
    x_full = np.array(grouped_data[sev])
    print(x_full.shape)
    print(x_full[0:10])

    # Get the train and validation sets
    # We will use the last 90 days for validation
    x_train = x_full[0:1371]
    print(x_train.shape)
    x_valid = x_full[1371:]
    print(x_valid.shape)

    np.save(f'training_data/daily_{sev}_train.npy', arr=x_train)
    np.save(f'training_data/daily_{sev}_valid.npy', arr=x_valid)
