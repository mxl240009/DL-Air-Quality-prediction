import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# 讀 EPA 資料
df = pd.read_csv("pm25_dallas.csv")

# 排序時間（一定要）
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

# 假設欄位叫 Arithmetic Mean（EPA 常見）
values = df["Daily Mean PM2.5 Concentration"].values

# 處理缺失值
values = pd.Series(values).ffill().values

# normalization
scaler = MinMaxScaler()
values = scaler.fit_transform(values.reshape(-1, 1))

# 建立時間序列
window_size = 7

X = []
y = []

for i in range(len(values) - window_size):
    X.append(values[i:i+window_size])
    y.append(values[i+window_size])

X = np.array(X)
y = np.array(y)

# reshape
X = X.reshape((X.shape[0], X.shape[1], 1))

# train/test split（時間序列不能 shuffle）
split = int(len(X) * 0.8)

X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 建模型（升級版）
model = Sequential()
model.add(LSTM(64, activation='relu', input_shape=(window_size, 1)))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

# 訓練
history = model.fit(
    X_train, y_train,
    epochs=20,
    validation_data=(X_test, y_test),
    verbose=1
)

# 預測
pred = model.predict(X_test)

# 還原數值
pred = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# 📊 畫圖（超重要）
plt.plot(y_test, label="Real")
plt.plot(pred, label="Prediction")
plt.legend()
plt.title("PM2.5 Prediction")
plt.show()