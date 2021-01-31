import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

Time_Series1 = np.random.laplace(scale=0.5, size=90)
Time_Series2 = np.random.laplace(scale=0.5, size=30) + 1.5
Time_Series3 = np.random.laplace(scale=0.5, size=30) + 0.7
Time_Series4 = np.random.laplace(scale=0.5, size=60) + 1
Time_Series5 = np.random.laplace(scale=0.5, size=30) + 0.9
# for i in range(0,len(Time_Series)):
#     if Time_Series[i] < 0:
#         Time_Series[i]=0
#     else:
#         Time_Series[i] = Time_Series[i]
TS1 = pd.Series(Time_Series1)
TS2 = pd.Series(Time_Series2)
TS3 = pd.Series(Time_Series3)
TS4 = pd.Series(Time_Series4)
TS5 = pd.Series(Time_Series5)
Time_Series = pd.concat([TS1, TS3, TS5, TS4, TS2, TS2], axis=0)
Time = pd.DatetimeIndex(Time_Series, start=2008)
time = datetime(year=2015, month=1, day=1)
dates = pd.date_range(start='2015-01-01', periods=len(Time_Series), freq='D')
data = {'TimeIndex': dates, 'Data' : Time_Series}
Data = pd.DataFrame(data=data)
Data.set_index('Time index', inplace=True)
plt.scatter(x=Data.index, y=Data)
plt.show()
# print(Data)
Data.to_csv('Data_production_tend_hauss.csv', sep=',')
