import pandas as pd
import numpy as np
from datetime import datetime

INP_FILE = 'data/patient_arrival_data.csv'


def parse_func(date_col):
    return [datetime.strptime(d, '%m/%d/%Y %H:%M') for d in date_col]


# STEP-1: read the patient arrival data and process the date column
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

# STEP-2: group data at daily level and do sanity checks
print('\nGroup data at a daily level (after dropping last 5 days\' data)')
grouped_data = data.groupby(['year', 'month', 'day'], as_index=False).agg({'ESI 1': 'sum', 'ESI 2': 'sum',
                                                                           'ESI 3': 'sum', 'ESI 4': 'sum',
                                                                           'ESI 5': 'sum', 'Total': 'sum'})
# Remove last 5 rows since we want the data to have full weeks
grouped_data.drop(grouped_data.tail(n=5).index, inplace=True)

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


# STEP-3: Split by ESIs and restructure the numpy array
for sev in ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5', 'Total']:
    print('\n**********************************')
    print(f'Patient severity: {sev}')
    # Convert data to numpy array
    # order=F implies that we fill up the data column-by-column
    # i.e., fill all the rows for column 0, then column 1, and so on
    # Therefore, each column represents a week with 7 data points
    # and each row's length will equal the total number of weeks
    x_full = np.array(grouped_data[sev])
    x_full = np.reshape(x_full, (7, -1), order='F')
    print(x_full.shape)
    print(x_full[0].shape)

    # Get the train and validation sets
    # We will use the last 4 weeks for validation
    x_train = x_full[:, 0:204]
    print(x_train.shape)
    x_valid = x_full[:, 204:]
    print(x_valid.shape)

    np.save(f'training_data/weekly_{sev}_train.npy', arr=x_train)
    np.save(f'training_data/weekly_{sev}_valid.npy', arr=x_valid)
