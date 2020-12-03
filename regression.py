import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as st
#%% Data
df = pd.read_excel("data/FinalDemand.xlsx")

#%% Splitting into training and testing

split_cut = 21215 # It is 0.85 aprox
train_set, test_set = df[0:split_cut], df[split_cut:len(df)]

#%% Perform regression
months = ['M%s' %(i) for i in range(1,12)]
hours = ['H%s' %(i) for i in range(1,24)]
#days =  ['D%s' %(i) for i in range(1,7)]
reg = sm.OLS(train_set['Actual'], sm.add_constant(train_set[hours + months])).fit()
print(reg.summary())

y_fit = reg.fittedvalues
y_pred = reg.predict(sm.add_constant(test_set[hours + months][0:744]))

#%% Save prediction on test dataframe
df_result = pd.DataFrame(test_set[0:744])
df_result['RegressionPred'] = y_pred.values
