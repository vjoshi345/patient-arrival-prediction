import pandas as pd
from datetime import datetime

INP_FILE = 'data/patient_arrival_data.csv'


def parse_func(date_col):
    return [datetime.strptime(d, '%m/%d/%Y %H:%M') for d in date_col]


# Read the patient arrival and process the Date column
data = pd.read_csv(INP_FILE, parse_dates=['Date'], date_parser=parse_func)
data['year'] = [d.year for d in data['Date']]
data['month'] = [d.month for d in data['Date']]
data['day'] = [d.day for d in data['Date']]
data['hour'] = [d.hour for d in data['Date']]
print(data.shape)
print(data.head())
print(data.dtypes)

# Check if the patient arrival is already sorted by time
data_sorted_by_cols = data.sort_values(by=['year', 'month', 'day', 'hour'], ascending=[True, True, True, True])
data_sorted_by_date = data.sort_values(by=['Date'], ascending=[True])

print('Comparing original data with data sorted by split columns: ', data_sorted_by_cols.equals(data))
print('Comparing original data with data sorted by \'Date\' column: ', data_sorted_by_date.equals(data))

# Count NULLs in each column
print("No. of empty rows in Date: ", sum((data['Date'] is None) | (data['Date'] == '')))
print("No. of empty rows in ESI 1: ", sum((data['ESI 1'] is None) | (data['ESI 1'] == '')))
print("No. of empty rows in ESI 2: ", sum((data['ESI 2'] is None) | (data['ESI 2'] == '')))
print("No. of empty rows in ESI 3: ", sum((data['ESI 3'] is None) | (data['ESI 3'] == '')))
print("No. of empty rows in ESI 4: ", sum((data['ESI 4'] is None) | (data['ESI 4'] == '')))
print("No. of empty rows in ESI 5: ", sum((data['ESI 5'] is None) | (data['ESI 5'] == '')))
print("No. of empty rows in Total: ", sum((data['Total'] is None) | (data['Total'] == '')))
