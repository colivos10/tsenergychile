#%% Imports
import pandas as pd
import matplotlib.pyplot as plt

#%% data
df = pd.read_excel("data/FinalDemand.xlsx")

#%% National electricity load full data
fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df['Date'], df['Actual'])
ax.set_title('National electricity load (MW)')
plt.savefig('plots/Nationalload.pdf', bbox_inches='tight')
plt.savefig('plots/Nationalload.png', bbox_inches='tight')
plt.show()

# %% National electricity load 1 week
end_date = '2019-05-13'
initial_date = '2019-05-06'

df1 = df[(df['Date'] < end_date) & (df['Date'] >= initial_date)]

fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df1['Date'], df1['Actual'])
ax.set_title('National electricity load (MW)')
plt.savefig('plots/Nationalload1week.pdf', bbox_inches='tight')
plt.savefig('plots/Nationalload1week.png', bbox_inches='tight')
plt.show()

#%% National electricity load 1 month
end_date = '2019-06-06'
initial_date = '2019-05-06'

df1 = df[(df['Date'] < end_date) & (df['Date'] >= initial_date)]

fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df1['Date'], df1['Actual'])
ax.set_title('National electricity load (MW)')
plt.savefig('plots/Nationalload1month.pdf', bbox_inches='tight')
plt.savefig('plots/Nationalload1month.png', bbox_inches='tight')
plt.show()

#%% National electricity load 6 month
end_date = '2019-11-06'
initial_date = '2019-05-06'

df1 = df[(df['Date'] < end_date) & (df['Date'] >= initial_date)]

fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df1['Date'], df1['Actual'])
ax.set_title('National electricity load (MW)')
plt.savefig('plots/Nationalload6month.pdf', bbox_inches='tight')
plt.savefig('plots/Nationalload6month.png', bbox_inches='tight')
plt.show()

#%% Importing results for plotting 1 day
df = pd.read_excel("results/prediction1weeks.xlsx")

#%% regression
fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df['Date'][0:24], df['Actual'][0:24], label='Actual demand', color='tab:blue', linewidth=3)
ax.plot(df['Date'][0:24], df['Forecasted'][0:24], label='ISO forecasted demand', color='tab:orange', linewidth=3)
ax.plot(df['Date'][0:24], df['Regression1'][0:24], label='Regression 1', color='tab:green', linewidth=3)
ax.plot(df['Date'][0:24], df['Regression2'][0:24], label='Regression 2', color='tab:cyan', linewidth=3)
ax.legend(loc=2)
ax.set_ylim(7000, 10500)
ax.set_title('National electricity load (MW)')
plt.savefig('plots/regoneday.pdf', bbox_inches='tight')
plt.savefig('plots/regoneday.png', bbox_inches='tight')
plt.show()
#%% neural network
fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df['Date'][0:24], df['Actual'][0:24], label='Actual demand', color='tab:blue', linewidth=3)
ax.plot(df['Date'][0:24], df['Forecasted'][0:24], label='ISO forecasted demand', color='tab:orange', linewidth=3)
ax.plot(df['Date'][0:24], df['NN-(32, 64)'][0:24], label='Neural network - Hidden layer: 32 Nodes ', color='tab:red', linewidth=3)
ax.plot(df['Date'][0:24], df['NN-(64, 64)'][0:24], label='Neural network - Hidden layer: 64 Nodes', color='tab:purple', linewidth=3)
ax.plot(df['Date'][0:24], df['NN-(128, 64)'][0:24], label='Neural network - Hidden layer: 128 Nodes', color='tab:olive', linewidth=3)
ax.legend(loc=2)
ax.set_ylim(7000, 10500)
ax.set_title('National electricity load (MW)')
plt.savefig('plots/nnoneday.pdf', bbox_inches='tight')
plt.savefig('plots/nnoneday.png', bbox_inches='tight')
plt.show()
#%% time series
fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df['Date'][0:24], df['Actual'][0:24], label='Actual demand', color='tab:blue', linewidth=3)
ax.plot(df['Date'][0:24], df['Forecasted'][0:24], label='ISO forecasted demand', color='tab:orange', linewidth=3)
ax.plot(df['Date'][0:24], df['TSAA24'][0:24], label='Holt winters - T:A S:M', color='tab:brown', linewidth=3)
ax.plot(df['Date'][0:24], df['TSAM24'][0:24], label='Holt winters - T:A S:A', color='tab:gray', linewidth=3)
ax.legend(loc=2)
ax.set_ylim(7000, 10500)
ax.set_title('National electricity load (MW)')
plt.savefig('plots/tsoneday.pdf', bbox_inches='tight')
plt.savefig('plots/tsoneday.png', bbox_inches='tight')
plt.show()

#%% all
fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df['Date'][0:24], df['Actual'][0:24], label='Actual demand', color='tab:blue', linewidth=3)
ax.plot(df['Date'][0:24], df['Forecasted'][0:24], label='ISO forecasted demand', color='tab:orange', linewidth=3)
ax.plot(df['Date'][0:24], df['Regression1'][0:24], label='Regression 1', color='tab:green', linewidth=3)
ax.plot(df['Date'][0:24], df['Regression2'][0:24], label='Regression 2', color='tab:cyan', linewidth=3)
ax.plot(df['Date'][0:24], df['TSAA24'][0:24], label='Holt winters - T:A S:M', color='tab:brown', linewidth=3)
ax.plot(df['Date'][0:24], df['TSAM24'][0:24], label='Holt winters - T:A S:A', color='tab:gray', linewidth=3)
ax.plot(df['Date'][0:24], df['NN-(32, 64)'][0:24], label='Neural network - Hidden layer: 32 Nodes ', color='tab:red', linewidth=3)
ax.plot(df['Date'][0:24], df['NN-(64, 64)'][0:24], label='Neural network - Hidden layer: 64 Nodes', color='tab:purple', linewidth=3)
ax.plot(df['Date'][0:24], df['NN-(128, 64)'][0:24], label='Neural network - Hidden layer: 128 Nodes', color='tab:olive', linewidth=3)
ax.legend(loc=2)
ax.set_ylim(7000, 10500)
ax.set_title('National electricity load (MW)')
plt.savefig('plots/alloneday.pdf', bbox_inches='tight')
plt.savefig('plots/alloneday.png', bbox_inches='tight')
plt.show()


################################
#%% Prediction 1 week
#%% regression
fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df['Date'], df['Actual'], label='Actual demand', color='tab:blue', linewidth=3)
ax.plot(df['Date'], df['Forecasted'], label='ISO forecasted demand', color='tab:orange', linewidth=3)
ax.plot(df['Date'], df['Regression1'], label='Regression 1', color='tab:green', linewidth=3)
ax.plot(df['Date'], df['Regression2'], label='Regression 2', color='tab:cyan', linewidth=3)
ax.legend(loc=2)
ax.set_ylim(7000, 10500)
ax.set_title('National electricity load (MW)')
plt.savefig('plots/regoneweek.pdf', bbox_inches='tight')
plt.savefig('plots/regoneweek.png', bbox_inches='tight')
plt.show()
#%% neural network
fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df['Date'], df['Actual'], label='Actual demand', color='tab:blue', linewidth=3)
ax.plot(df['Date'], df['Forecasted'], label='ISO forecasted demand', color='tab:orange', linewidth=3)
ax.plot(df['Date'], df['NN-(32, 64)'], label='Neural network - Hidden layer: 32 Nodes ', color='tab:red', linewidth=3)
ax.plot(df['Date'], df['NN-(64, 64)'], label='Neural network - Hidden layer: 64 Nodes', color='tab:purple', linewidth=3)
ax.plot(df['Date'], df['NN-(128, 64)'], label='Neural network - Hidden layer: 128 Nodes', color='tab:olive', linewidth=3)
ax.legend(loc=2)
ax.set_ylim(7000, 10500)
ax.set_title('National electricity load (MW)')
plt.savefig('plots/nnoneweek.pdf', bbox_inches='tight')
plt.savefig('plots/nnoneweek.png', bbox_inches='tight')
plt.show()
#%% time series
fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df['Date'], df['Actual'], label='Actual demand', color='tab:blue', linewidth=3)
ax.plot(df['Date'], df['Forecasted'], label='ISO forecasted demand', color='tab:orange', linewidth=3)
ax.plot(df['Date'], df['TSAA24'], label='Holt winters - T:A S:M', color='tab:brown', linewidth=3)
ax.plot(df['Date'], df['TSAM24'], label='Holt winters - T:A S:A', color='tab:gray', linewidth=3)
ax.legend(loc=2)
ax.set_ylim(7000, 10500)
ax.set_title('National electricity load (MW)')
plt.savefig('plots/tsoneweek.pdf', bbox_inches='tight')
plt.savefig('plots/tsoneweek.png', bbox_inches='tight')
plt.show()

#%% all
fig, ax = plt.subplots(1, figsize=(16,9))
ax.set_ylabel('Megawatt')
ax.plot(df['Date'], df['Actual'], label='Actual demand', color='tab:blue', linewidth=3)
ax.plot(df['Date'], df['Forecasted'], label='ISO forecasted demand', color='tab:orange', linewidth=3)
ax.plot(df['Date'], df['Regression1'], label='Regression 1', color='tab:green', linewidth=3)
ax.plot(df['Date'], df['Regression2'], label='Regression 2', color='tab:cyan', linewidth=3)
ax.plot(df['Date'], df['TSAA24'], label='Holt winters - T:A S:M', color='tab:brown', linewidth=3)
ax.plot(df['Date'], df['TSAM24'], label='Holt winters - T:A S:A', color='tab:gray', linewidth=3)
ax.plot(df['Date'], df['NN-(32, 64)'], label='Neural network - Hidden layer: 32 Nodes ', color='tab:red', linewidth=3)
ax.plot(df['Date'], df['NN-(64, 64)'], label='Neural network - Hidden layer: 64 Nodes', color='tab:purple', linewidth=3)
ax.plot(df['Date'], df['NN-(128, 64)'], label='Neural network - Hidden layer: 128 Nodes', color='tab:olive', linewidth=3)
ax.legend(loc=2)
ax.set_ylim(7000, 10500)
ax.set_title('National electricity load (MW)')
plt.savefig('plots/alloneweek.pdf', bbox_inches='tight')
plt.savefig('plots/alloneweek.png', bbox_inches='tight')
plt.show()

#%% Prediction 1 month
