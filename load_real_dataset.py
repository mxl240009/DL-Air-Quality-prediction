import pandas as pd

df = pd.read_csv("pm25_dallas.csv")

print(df.head())
print(df.columns)
print(len(df))