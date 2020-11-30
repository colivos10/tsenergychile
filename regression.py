import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

#%% Data
df = pd.read_excel("data/FinalData.xlsx")
#df = pd.concat([df, pd.get_dummies(df['Month']), pd.get_dummies(df['Hour'])], axis=1)

#%% Selecting part of the data

end_date = '2018-01-31'
initial_date = '2018-01-01'

df1 = df[(df['Date'] <= end_date) & (df['Date'] >= initial_date)]
df2 = df[(df['Date'] <= end_date) & (df['Date'] >= initial_date)]

y_train = df1['Actual']
#y_train = df1
#%%
months = ['M%s' %(i) for i in range(1,12)]
hours = ['H%s' %(i) for i in range(1,24)]
reg = sm.OLS(y_train, sm.add_constant(df1[hours])).fit()
reg.summary()

y_fit = reg.fittedvalues


#%%
plt.plot(y_train)
plt.plot(y_fit)
plt.show()

 #%% Measures
mse = reg.mse_model
mae = mean_absolute_error(y_train, y_fit)
aic = reg.aic
bic = reg.bic