import os
from pathlib import Path

import pandas as pd

year = '2019'  # please specify the year, our experiments use 2019

CURRENT_DIR = Path(__file__).resolve().parent
RAW_DATA_FILEPATH = CURRENT_DIR.parent.parent / 'data' / 'raw' / f'ca_his_raw_{year}.h5'
PROCESSED_FILEPATH = CURRENT_DIR / f'ca_his_{year}.h5'

ca_his = pd.read_hdf(RAW_DATA_FILEPATH)

### please comment this line if you don't want to do resampling
ca_his = ca_his.resample('15T').mean().round(0)
###

ca_his = ca_his.fillna(0)
print('check null value number', ca_his.isnull().any().sum())

ca_his.to_hdf(PROCESSED_FILEPATH, key='t', mode='w')