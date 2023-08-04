import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

ipdata = pd.read_csv("<historical data as csv for training>")
#if needed for finiding optimal regression line
#x = ipdata['Date']
#y = ipdata['Open']
#plt.scatter(x,y,)
tr = ipdata.iloc[:, 1:2].values
sc = MinMaxScaler(feature_range = (0, 1))
tr_sc = sc.fit_transform(tr)

X_tr = []
Y_tr = []
for i in range(60, len(tr_sc)):
    X_tr.append(tr_sc[i-60:i, 0])
    Y_tr.append(tr_sc[i, 0])
X_tr, Y_tr = np.array(X_tr), np.array(Y_tr)

# reshape data
X_tr = np.reshape(X_tr, ((X_tr.shape[0], X_tr.shape[1], 1)))
# create and fit the LSTM network
lreg = Sequential()
lreg.add(LSTM(units = 50, return_sequences = True, input_shape = (X_tr.shape[1], 1)))
lreg.add(Dropout(0.2))
lreg.add(LSTM(units = 50, return_sequences = True))
lreg.add(Dropout(0.2))
lreg.add(LSTM(units = 50, return_sequences = True))
lreg.add(Dropout(0.2))
lreg.add(LSTM(units = 50))
lreg.add(Dropout(0.2))
#o/p layer
lreg.add(Dense(units = 1))
#global minima point finding
lreg.compile(optimizer = 'adam', loss = 'mean_squared_error')
lreg.fit(X_tr, Y_tr, epochs = 1, batch_size = 32)

td = pd.read_csv("testing data as csv")
rsp = td.iloc[:, 1:2].values
dt = pd.concat((ipdata['Open'], td['Open']), axis = 0)
ip = dt[len(dt) - len(td) - 60:].values
ip = ip.reshape(-1,1)
ip = sc.transform(ip)
test = []
#since time stamp is 6o 
for i in range(60, len(ip)):
    test.append(ip[i-60:i, 0])
test = np.array(test)
test = np.reshape(test, (test.shape[0], test.shape[1], 1))  

psp = lreg.predict(test)  
psp = sc.inverse_transform(psp)

#plot
x = td['Date']
y = td['Open']
plt.scatter(x,y, label = "Real Stock Price")
plt.plot(psp, color = 'red', label = 'Calculated Stock Price')
plt.title(' Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel(' Stock Price')

plt.xticks(x, rotation='45')
plt.legend()
plt.show()

