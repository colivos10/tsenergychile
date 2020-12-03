import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
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


#%% Data
df = pd.read_excel("data/FinalDemand.xlsx")

#%% Splitting into training and testing

split_cut = 21215 # It is 0.85 aprox
train_set, test_set = df[0:split_cut-24], df[split_cut-24:len(df)]
look_back = 24 #seasonality

#%%
scaler = MinMaxScaler(feature_range=(0, 1))
train_norm = scaler.fit_transform(np.reshape(train_set['Actual'].values,(len(train_set),1)))
test_norm = scaler.fit_transform(np.reshape(test_set['Actual'].values,(len(test_set),1)))

#%%
#convert dataset into right shape in order to input into the DNN
train_x, train_y = convert2matrix(train_norm, look_back)
test_x, test_y = convert2matrix(test_norm, look_back)

#%% reshaping
train_x = np.reshape(train_x,(len(train_x),24))
test_x = np.reshape(test_x,(len(test_x),24))

#%%
df_result = pd.DataFrame(test_set[24:192])
n_epochs = [32, 64, 128]
n_hidden = [32, 64, 128]

for i in n_hidden:
    def model_dnn(look_back):
        model = Sequential()
        model.add(Dense(units=i, input_dim=look_back, activation='sigmoid'))
        # model.add(Dense(8, activation='sigmoid'))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
        return model
    for j in n_epochs:
        model=model_dnn(24)
        history=model.fit(train_x,train_y, epochs=j, verbose=1)
        prediction = scaler.inverse_transform(model.predict(train_x[21167-168:21167]))
        df_result[f'NN-{i,j}'] = prediction

df_result.to_excel('NNResults.xlsx')

