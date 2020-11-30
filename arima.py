import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.style.use('fivethirtyeight')
import pandas as pd
import statsmodels.api as sm
import matplotlib
#matplotlib.rcParams['axes.labelsize'] = 14
#matplotlib.rcParams['xtick.labelsize'] = 12
#matplotlib.rcParams['ytick.labelsize'] = 12
#matplotlib.rcParams['text.color'] = 'G'
#%% Data
df = pd.read_excel("data/Demand.xlsx")
df = pd.concat([df, pd.get_dummies(df['Month']), pd.get_dummies(df['Hour'])], axis=1)

#%% Selecting part of the data

end_date = '2019-02-01'
initial_date = '2019-01-01'

df1 = df[(df['Date'] < end_date) & (df['Date'] > initial_date)]

#%% Number of predictions
n_forecasted = 24

#%% Splitting data

df1_train = df1['Actual'][:-n_forecasted]
df1_test = df1['Actual'][-n_forecasted:]

#%%
from pylab import rcParams
rcParams['figure.figsize'] = 18, 8
decomposition = sm.tsa.seasonal_decompose(df1_train.values, model='additive', period=24)
fig = decomposition.plot()
plt.show()

#%% Sarima model
p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]
print('Examples of parameter for SARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

#%%
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df1_train,order=param,seasonal_order=param_seasonal,enforce_stationarity=False,enforce_invertibility=False)
            results = mod.fit()
            print('ARIMA{}x{}12 - AIC:{}'.format(param,param_seasonal,results.aic))
        except:
            continue

#%%
mod = sm.tsa.statespace.SARIMAX(df1_train.values,
                                order=(0, 0, 1),
                                seasonal_order=(1, 1, 1, 24),
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
print(results.summary().tables[1])

#%%
results.plot_diagnostics(figsize=(18, 8))
plt.show()

#%% Autocorrelation plot
autocorrelation_plot(df1_train.values)
plt.show()

#%%
model = ARIMA(df1_train, order=(2,1,2))
results = model.fit(disp=-1)
#plt.plot(df_log_shift)
plt.plot(results.fittedvalues, color='red')