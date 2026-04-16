import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 你 STEP 3 的模擬資料
values = [27, 30, 28, 35, 33, 31, 29, 40, 38, 36]

window_size = 3

X = []
y = []

for i in range(len(values) - window_size):
    X.append(values[i:i+window_size])
    y.append(values[i+window_size])

X = np.array(X)
y = np.array(y)

# LSTM 要 3D input: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 建模型
model = Sequential()

model.add(LSTM(32, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# 訓練
model.fit(X, y, epochs=100, verbose=1)

# 預測最後一組
test_input = np.array(values[-3:]).reshape((1, 3, 1))
prediction = model.predict(test_input)

print("Prediction:", prediction[0][0])