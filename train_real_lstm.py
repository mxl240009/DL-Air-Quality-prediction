import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 讀資料
df = pd.read_csv("pm25_history.csv")
values = df["avg"].values

# 再做 normalization
scaler = MinMaxScaler()
values = scaler.fit_transform(values.reshape(-1, 1))

# window
window_size = 7 

X = []
y = []

for i in range(len(values) - window_size):
    X.append(values[i:i+window_size])
    y.append(values[i+window_size])

X = np.array(X)
y = np.array(y)

# reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# model
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(7, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# train
history = model.fit(
    X_train, y_train,
    epochs=50,
    validation_data=(X_test, y_test),
    verbose=1
)

# prediction
pred = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

print("\n=== Predictions ===")
print(pred[:5].flatten())

print("\n=== Real ===")
print(y_test[:5])