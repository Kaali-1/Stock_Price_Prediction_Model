# Stock Prediction Model
# Anshmeet Singh

# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.python.keras.layers import LSTM, Dense
from tensorflow.python.keras.models import Sequential

# Read CSV File - Insert Stock Ticker & Download CSV
data = pd.read_csv('TSLA.csv', header=0, usecols=['Date', 'Close'], parse_dates=True, index_col=['Date'])
plt.plot(data['Close'])

# Graph Scale
scaler = MinMaxScaler()
scaledData = scaler.fit_transform(data)

# Datasets
trainLength = int(len(scaledData) * 0.7)
testLength = len(scaledData) - trainLength
trainData = scaledData[0:trainLength, :]
testData = scaledData[trainLength:len(scaledData), :]


def create_dataset(dataset, timestep):
    data_x, data_y = [], []
    for i in range(len(dataset) - timestep - 1):
        data_x.append(dataset[i:(i + timestep), 0])
        data_y.append(dataset[i + timestep, 0])
    return np.array(data_x), np.array(data_y)


timestep = 1
trainX, trainY = create_dataset(trainData, timestep)
testX, testY = create_dataset(testData, timestep)

trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Prediction Model
model = Sequential()
model.add(LSTM(255, input_shape=(1, 1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='mean_squared_error', optimizer='adam', metrix=['accuracy'])

# Model Training
model.fit(trainX, trainY, epochs=1, bathc_size=1, verbose=1)
score = model.evaluate(trainX, trainY, verbose=0)
print('Model Loss = ', score[0])
print('Model Accuracy = ', score[1])

# Prediction
trainPredictions = model.predict(trainX)
testPredictions = model.predict(testX)

# Transform Data
trainPredictions = scaler.inverse_transform(trainPredictions)
trainY = scaler.inverse_transform([trainY])
testPredictions = scaler.inverse_transform(testPredictions)
testX = scaler.inverse_transform([testY])

# Plotting Data Sets
trainPredictPlot = np.empty_like(scaledData)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[1:len(trainPredictions) + 1, :] = trainPredictions

testPredictPlot = np.empty_like(scaledData)
testPredictPlot[:, :] = np.nan
testPredictPlot[1:len(trainPredictions) + 2 + 1:len(scaledData) - 1, :] = testPredictions

plt.plot(scaler.inverse_transform(scaledData))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()