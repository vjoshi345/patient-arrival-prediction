from collections import Counter
import pandas as pd

INP_FILE = 'data/patient_arrival_data.csv'

data = pd.read_csv(INP_FILE)
# print(data.shape)
print(data.head())
# print(data.dtypes)

# # Count NULLs in each column
# print("No. of empty rows in Date: ", sum((data['Date'] is None) | (data['Date'] == '')))
# print("No. of empty rows in ESI 1: ", sum((data['ESI 1'] is None) | (data['ESI 1'] == '')))
# print("No. of empty rows in ESI 2: ", sum((data['ESI 2'] is None) | (data['ESI 2'] == '')))
# print("No. of empty rows in ESI 3: ", sum((data['ESI 3'] is None) | (data['ESI 3'] == '')))
# print("No. of empty rows in ESI 4: ", sum((data['ESI 4'] is None) | (data['ESI 4'] == '')))
# print("No. of empty rows in ESI 5: ", sum((data['ESI 5'] is None) | (data['ESI 5'] == '')))
# print("No. of empty rows in Total: ", sum((data['Total'] is None) | (data['Total'] == '')))

len_series = data['Date'].str.len()
print(type(len_series))
print(len_series.shape)
len_cnt = Counter(len_series)
print(len_cnt)
