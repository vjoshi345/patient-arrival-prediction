import pandas as pd
import numpy as np

INP_FILE = 'data/patient_arrival_data.csv'

print('Running for input file: ', INP_FILE.split('/')[-1])
data = pd.read_csv(INP_FILE, dtype=object)
print(data.shape)
print(data.dtypes)

for sev in ['ESI 1', 'ESI 2', 'ESI 3', 'ESI 4', 'ESI 5', 'Total']:
    print('\n**********************************')
    print(f'Patient severity: {sev}')
    # Convert data to numpy array
    # order=F implies that we fill up the data column-by-column
    # i.e., fill all the rows for column 0, then column 1, and so on
    # Therefore, each column represents a day with 24 data points
    x_full = np.array(data[sev])
    x_full = np.reshape(x_full, (24, -1), order='F')
    print(x_full.shape)
    print(x_full[0].shape)

    # Get the train and validation sets
    # We will use the last 90 days for validation
    x_train = x_full[:, 0:1371]
    print(x_train.shape)
    x_valid = x_full[:, 1371:]
    print(x_valid.shape)

    np.save(f'training_data/hourly_{sev}_train.npy', arr=x_train)
    np.save(f'training_data/hourly_{sev}_valid.npy', arr=x_valid)
