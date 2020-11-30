import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels
#%% Data
df = pd.read_excel("data/FinalDemand.xlsx")

#%% Selecting part of the data
#end_date = '2020-01-01'
#initial_date = '2018-01-01'

#df1 = df[(df['Date'] < end_date) & (df['Date'] >= initial_date)]
df1 = df[['Date','Actual']]
#%% Splitting into training and testing
#df1 = df[['Date','Actual']]
split_size = 0.85
split_int = round(split_size*len(df))
train_set, test_set = df[0:split_int], df[split_int:len(df)]

#%%
months = ['M%s' %(i) for i in range(1,12)]
hours = ['H%s' %(i) for i in range(1,24)]
reg = sm.OLS(train_set['Actual'], sm.add_constant(train_set[hours + months])).fit()
reg.summary()

y_fit = reg.fittedvalues
y_pred = reg.predict(sm.add_constant(test_set[hours + months]))
#%%
plt.plot(train_set['Actual'])
plt.plot(train_set['Forecasted'], alpha=0.6)
plt.plot(y_fit, alpha=0.6)
plt.show()

#%%
plt.plot(test_set['Actual'])
plt.plot(test_set['Forecasted'], alpha=0.6)
plt.plot(y_pred, alpha=0.6)
plt.show()

#%% Measures training
mse_train = mean_squared_error(train_set['Actual'], y_fit)
mae_train = mean_absolute_error(train_set['Actual'], y_fit)
aic_train = reg.aic
bic_train = reg.bic

#%% Measures testing
mae_test = mean_absolute_error(test_set['Actual'], y_pred)
mse_test = mean_squared_error(test_set['Actual'], y_pred)


