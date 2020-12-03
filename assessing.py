import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
#%% Importing results for plotting
df = pd.read_excel("results/prediction1weeks.xlsx")

#%% Columns
col_list = df.columns

#%% 1 week
for method in range(2,len(col_list)):
    #print(col_list[method])
    #print(mean_squared_error(df['Actual'], df[col_list[method]]))
    print(mean_absolute_error(df['Actual'], df[col_list[method]]))

#%% 1 day
for method in range(2,len(col_list)):
    #print(col_list[method])
    #print(mean_squared_error(df['Actual'][0:24], df[col_list[method]][0:24]))
    print(mean_absolute_error(df['Actual'][0:24], df[col_list[method]][0:24]))