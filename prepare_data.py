import pandas as pd
import numpy as np

# 讀資料
df = pd.read_csv("waqi_data.csv")

# 這裡先模擬時間序列（因為你目前只有1筆）
# 真實版本之後會換 API 多時間資料
values = [27, 30, 28, 35, 33, 31, 29, 40, 38, 36]

df = pd.DataFrame(values, columns=["pm25"])

# 建立 X (過去3筆) → y (下一筆)
X = []
y = []

window_size = 3

for i in range(len(df) - window_size):
    X.append(df["pm25"].iloc[i:i+window_size].values)
    y.append(df["pm25"].iloc[i+window_size])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nX sample:")
print(X[:3])

print("\ny sample:")
print(y[:3])