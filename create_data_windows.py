import pandas as pd
from datetime import datetime
from collections import Counter

INP_FILE = 'data/patient_arrival_data.csv'


def parse_func(date_col):
    return [datetime.strptime(d, '%m/%d/%Y %H:%M') for d in date_col]


# Read the patient arrival and process the Date column
print('Reading hourly patient arrival data:')
data = pd.read_csv(INP_FILE, parse_dates=['Date'], date_parser=parse_func)
data['year'] = [d.year for d in data['Date']]
data['month'] = [d.month for d in data['Date']]
data['day'] = [d.day for d in data['Date']]
data['hour'] = [d.hour for d in data['Date']]
print(data.shape)
print(data.head())
print(data.dtypes)

# Check if the patient arrival is already sorted by time
print('\nChecking if patient arrival data is already sorted by time:')
data_sorted_by_cols = data.sort_values(by=['year', 'month', 'day', 'hour'], ascending=[True, True, True, True])
data_sorted_by_date = data.sort_values(by=['Date'], ascending=[True])

print('Comparing original data with data sorted by split columns: ', data_sorted_by_cols.equals(data))
print('Comparing original data with data sorted by \'Date\' column: ', data_sorted_by_date.equals(data))

# Generate week and 30-day identifiers
print('\nGenerating week and 30-day identifiers')
nrows = data.shape[0]

week_id = [int(i/168) for i in range(nrows)]
week_cnt = Counter(week_id)
print(week_cnt)
data['week_id'] = week_id

thirty_day_id = [int(i/720) for i in range(nrows)]
thirty_day_cnt = Counter(thirty_day_id)
print(thirty_day_cnt)
data['thirty_day_id'] = thirty_day_id

# Groupby and sum to create data by different time windows
print('\nGroupby and sum up weekly data:')
agg_dict = {'ESI 1': 'sum', 'ESI 2': 'sum', 'ESI 3': 'sum', 'ESI 4': 'sum', 'ESI 5': 'sum',
            'Total': 'sum'}
data_weekly_filtered = data.groupby(['week_id'], as_index=False).filter(lambda x: len(x) == 168)
data_weekly_final = data_weekly_filtered.groupby(['week_id'], as_index=False).agg(agg_dict)
print(data_weekly_final.shape)
print(data_weekly_final.head())
data_weekly_final.to_csv('data/patient_arrival_weekly.csv', index=False)

print('\nGroupby and sum up 30-day data:')
data_thirty_day_filtered = data.groupby(['thirty_day_id'], as_index=False).filter(lambda x: len(x) == 720)
data_thirty_day_final = data_thirty_day_filtered.groupby(['thirty_day_id'], as_index=False).agg(agg_dict)
print(data_thirty_day_final.shape)
print(data_thirty_day_final.head())
data_thirty_day_final.to_csv('data/patient_arrival_30day.csv', index=False)
