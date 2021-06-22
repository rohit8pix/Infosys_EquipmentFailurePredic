import io
import pandas as pd 
df2 = pd.read_csv(io.BytesIO(uploaded['ALLtrainMescla5D.csv']))
#df_test = pd.read_csv(io.BytesIO(uploaded['ALLtestMescla5D.csv']))
from pandas import DataFrame
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
%matplotlib inline 
sns.set(color_codes=True)
from sklearn.preprocessing import MinMaxScaler



#OUTPUT 1
df2.dtypes
max_cycle_data = df2.groupby(['machineID'], sort=False)['time_in_cycles'].max().reset_index().rename(columns={'time_in_cycles':'MaxCycleID'})
max_cycle_data
merge_data = pd.merge(df2, max_cycle_data, how='inner', on='machineID')
merge_data['RUL'] = merge_data['MaxCycleID']-merge_data['time_in_cycles']
merge_data[['time_in_cycles','RUL']]
#OUTPUT 2
list(merge_data)
len(merge_data[merge_data['RUL'] == 0])
len(merge_data['machineID'].unique())

# DATA PRROCESSING & EDA
axes = merge_data.describe().T.plot.bar(subplots=True, figsize=(15,12))
constant_col = [ col for col in merge_data.columns if len(merge_data[col].unique()) <= 3 ]
print('Columns with constant values: \n' + str(constant_col) + '\n')
info_cols = ['machineID','time_in_cycles', 'RUL']
sensor_cols = ['voltmean_24h','rotatemean_24h','pressuremean_24h','vibrationmean_24h','voltsd_24h','rotatesd_24h','pressuresd_24h','vibrationsd_24h','voltmean_5d','rotatemean_5d',
               'pressuremean_5d','vibrationmean_5d','voltsd_5d','rotatesd_5d','pressuresd_5d','vibrationsd_5d','error1','error2','error3','error4','error5','comp1','comp2','comp3',
               'comp4','model','age','DI','RULWeek','failure','failed', 'MaxCycleID']

data_corr = merge_data[ sensor_cols].corr(method='pearson')
fig, ax = plt.subplots(figsize=(15,12))
sns.heatmap(data_corr, linewidths=.5)
check_Data = merge_data[info_cols + sensor_cols].plot(subplots=True, figsize=(19, 19))
machine_series_length = merge_data.groupby(['machineID'], sort=False)['time_in_cycles'].max().sort_values()
shortest_machine = machine_series_length.index[0]
longest_machine = machine_series_length.index[-1]

print("The machine with the shortest cycles is no. {} with {} cycles lifetime and the one with the longest cycles run is no. {} running {} cycles.".\
      format(shortest_machine, machine_series_length.iloc[0], longest_machine, machine_series_length.iloc[-1]))
Train_Data = merge_data[['machineID','time_in_cycles','voltmean_24h','rotatemean_24h','pressuremean_24h','vibrationmean_24h','voltsd_24h','rotatesd_24h','pressuresd_24h','vibrationsd_24h','voltmean_5d','rotatemean_5d',
              'pressuremean_5d','vibrationmean_5d','voltsd_5d','rotatesd_5d','pressuresd_5d','vibrationsd_5d',
            'age','DI','RUL']]
Train_Data_copy = Train_Data.copy()
Period = 30
Train_Data_copy['fail_label'] = Train_Data_copy['RUL'].apply(lambda x: 1 if x <= Period else 0) 
Train_Data_copy.head(10)
