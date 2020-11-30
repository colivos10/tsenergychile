import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.preprocessing import MinMaxScaler
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import itertools
import warnings
warnings.filterwarnings('ignore')

#%% Data function
def convert2matrix(data_arr, look_back):
 X, Y =[], []
 for i in range(len(data_arr)-look_back):
  d=i+look_back
  X.append(data_arr[i:d])
  Y.append(data_arr[d])
 return np.array(X), np.array(Y)

def model_dnn(look_back):
    model=Sequential()
    model.add(Dense(units=32, input_dim=look_back, activation='sigmoid'))
    model.add(Dense(8, activation='sigmoid'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',  optimizer='adam',metrics = ['mse', 'mae'])
    return model

 def model_loss(history):
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Test Loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.legend(loc='upper right')
    plt.show();

def prediction_plot(testY, test_predict):
  len_prediction=[x for x in range(len(testY))]
  plt.figure(figsize=(8,4))
  plt.plot(len_prediction, testY[:l], marker='.', label="actual")
  plt.plot(len_prediction, test_predict[:l], 'r', label="prediction")
  plt.tight_layout()
  plt.subplots_adjust(left=0.07)
  plt.ylabel('Ads Daily Spend', size=15)
  plt.xlabel('Time step', size=15)
  plt.legend(fontsize=15)
  plt.show();


#%% import data
df = pd.read_excel("data/FinalData.xlsx")
end_date = '2018-03-31'
initial_date = '2018-01-01'

df1 = df[(df['Date'] <= end_date) & (df['Date'] >= initial_date)]
#df2 = df[(df['Date'] <= end_date) & (df['Date'] >= initial_date)]
df2 = df1['Actual']
#%%
#Split data set into testing dataset and train dataset
train_size = 200
train, test = df2.values[train_size:], df2.values[len(df2)-train_size: len(df2)]
# setup look_back window
look_back = 24

#%%
scaler = MinMaxScaler(feature_range=(0, 1))#LTSM is senstive to the scale of features
train_norm = scaler.fit_transform(np.reshape(train,(len(train),1)))
test_norm = scaler.fit_transform(np.reshape(test,(len(test),1)))

#%%
#convert dataset into right shape in order to input into the DNN
trainX, trainY = convert2matrix(train_norm, look_back)
testX, testY = convert2matrix(test_norm, look_back)

#%% reshaping
trainX = np.reshape(trainX,(len(trainX),24))
#trainY = np.reshape(trainY,(len(trainY),24))
testX = np.reshape(testX,(len(testX),24))
#testY = np.reshape(testY,(len(testY),24))

#%%
model=model_dnn(24)
history=model.fit(trainX,trainY, epochs=70, batch_size=30, verbose=1)

#%% Prediction

#predictions = model.predict(testX)
plt.plot(testY)
plt.plot(model.predict(testX))
plt.show()

#%%
train_score = model.evaluate(trainX, trainY, verbose=0)
print('Train Root Mean Squared Error(RMSE): %.2f; Train Mean Absolute Error(MAE) : %.2f '
% (np.sqrt(train_score[1]), train_score[2]))
test_score = model.evaluate(testX, testY, verbose=0)
print('Test Root Mean Squared Error(RMSE): %.2f; Test Mean Absolute Error(MAE) : %.2f '
% (np.sqrt(test_score[1]), test_score[2]))
model_loss(history)

#%%
plt.plot(testY)
plt.plot(model.predict(testX))
plt.show()


