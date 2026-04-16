import pandas as pd
import numpy as np

df = pd.read_csv("pm25_history.csv")

# 只用 avg（先簡化問題）
values = df["avg"].values

window_size = 3

X = []
y = []

for i in range(len(values) - window_size):
    X.append(values[i:i+window_size])
    y.append(values[i+window_size])

X = np.array(X)
y = np.array(y)

# LSTM 需要 3D
X = X.reshape((X.shape[0], X.shape[1], 1))

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nX sample:")
print(X[:2])

print("\ny sample:")
print(y[:2])