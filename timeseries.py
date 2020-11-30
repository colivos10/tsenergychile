#%% Imports
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.seasonal import seasonal_decompose

#%% Data
df = pd.read_excel("data/Demand.xlsx")
df = pd.concat([df, pd.get_dummies(df['Month']), pd.get_dummies(df['Hour'])], axis=1)

#%% Selecting part of the data

end_date = '2019-12-01'
initial_date = '2019-01-01'

df1 = df[(df['Date'] < end_date) & (df['Date'] > initial_date)]

#%% Number of predictions
n_forecasted = 24

#%% Splitting data

df1_train = df1['Actual'][:-n_forecasted]
df1_test = df1['Actual'][-n_forecasted:]

#%% Parameters
alpha = [i/10 for i in range(1,10)]
beta = [i/10 for i in range(1,10)]
gamma = [i/10 for i in range(1,10)]

trend_type = ['add', 'mul']
seasonal_type = ['add', 'mul']

#%% Fitting model

sse_list = []
aic_list = []
bic_list = []

rows = []
for alpha_it in alpha:
    for beta_it in beta:
        for gamma_it in gamma:
            for trend_it in trend_type:
                for seasonal_it in seasonal_type:
                    model = HWES(df1_train, seasonal_periods=24*7, trend=trend_it, seasonal=seasonal_it)
                    fitted = model.fit(smoothing_level=alpha_it, smoothing_slope=beta_it, smoothing_seasonal=gamma_it)
                    rows.append([alpha_it, beta_it, gamma_it, trend_it, seasonal_it, fitted.sse, fitted.aic, fitted.bic, fitted.aicc])

#%%
df2 = pd.DataFrame(rows, columns=['alpha', 'beta', 'gamma', 'trend', 'seasonal', 'sse', 'aic', 'bic', 'aicc'])
df2.to_excel('results.xlsx')
#%%
model = HWES(df1_train, seasonal_periods=24*7, seasonal='add', trend='add')
fitted = model.fit(smoothing_level=0.8, smoothing_slope=0.1, smoothing_seasonal=0.1)
print(fitted.summary())
demand_forecast = fitted.forecast(steps=24)
fig = plt.figure(figsize=(6,4))
fig.suptitle('Energy demand in January 2019')
past = plt.plot(df1_train.index, df1_train, 'b.-', label='Demand history', alpha=0.6)
#future = plt.plot(df1_test.index, df1_test, 'r.-', label='Actual demand')
predicted_future = plt.plot(df1_test.index, demand_forecast, 'g.-', label='Demand forecast')
#predicted_iso = plt.plot(df1.index, df1['Forecasted'], 'y.-', label='ISO forecast', alpha=0.2)
#plt.legend(handles=[past, future, predicted_future, predicted_iso])
plt.savefig('forecasted.pdf')
plt.show()

#%% Decomposition
fig = plt.figure(figsize=(6,4))
decomp = seasonal_decompose(df1['Actual'], model='multiplicative')
decomp.plot()
plt.show()

#%%
import statsmodels.api as sm
res = sm.tsa.seasonal_decompose(df1_train, period=24*7)
resplot = res.plot()
resplot.show()