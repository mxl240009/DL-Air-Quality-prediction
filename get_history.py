import requests
import pandas as pd

TOKEN = "4c3ea47ca4c71c2ddfa6b7ba6348f6a84eca4a25"

city = "dallas"

url = f"https://api.waqi.info/feed/{city}/?token={TOKEN}"

response = requests.get(url)
data = response.json()

if data["status"] != "ok":
    print("API error:", data)
    exit()

# 取得歷史資料（注意：WAQI 有時只有最近 snapshots）
history = data["data"].get("forecast", {}).get("daily", {}).get("pm25", [])

df = pd.DataFrame(history)

print(df.head())

df.to_csv("pm25_history.csv", index=False)