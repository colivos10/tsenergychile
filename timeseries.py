#%% Imports
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

#%% Data
df = pd.read_excel("data/FinalDemand.xlsx")

#%% Splitting into training and testing

split_cut = 21215 # It is 0.85 aprox
train_set, test_set = df[0:split_cut], df[split_cut:len(df)]

#%% Number of predictions
n_forecasted = 168

#%% Parameters

trend_type = ['add', 'mul']
seasonal_type = ['add', 'mul']
seasonal_size = [24, 24*7]

#%% Forecasting
measures = []

df_result = pd.DataFrame()

for i in trend_type:
    for j in seasonal_type:
        for k in seasonal_size:
            model = HWES(train_set['Actual'], seasonal_periods=k, trend=i, seasonal=j)
            fitted = model.fit(optimized=True, remove_bias=True)
            print(fitted.summary())
            measures.append((i, j, k, model.params['smoothing_level'], model.params['smoothing_slope'], model.params['smoothing_seasonal'], fitted.sse, fitted.aic, fitted.bic))

            demand_forecast = fitted.forecast(steps=168)
            df_result[f'TS{i,j, k}'] = demand_forecast

df_measures = pd.DataFrame(measures, columns=['trend', 'seasonal','size', 'alpha', 'beta', 'gamma', 'sse', 'aic', 'bic'])

#%%
df_result.to_excel('tsresult1.xlsx')
df_measures.to_excel('measurests.xlsx')
