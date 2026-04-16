import requests
import pandas as pd

TOKEN = "4c3ea47ca4c71c2ddfa6b7ba6348f6a84eca4a25"

city = "dallas"

url = f"https://api.waqi.info/feed/{city}/?token={TOKEN}"

response = requests.get(url)
data = response.json()

print(data)  # 先看回傳內容

if data["status"] != "ok":
    print("API error")
    exit()

# WAQI 結構
iaqi = data["data"]["iaqi"]

pm25 = iaqi.get("pm25", {}).get("v", None)

result = pd.DataFrame([{
    "city": city,
    "pm25": pm25
}])

result.to_csv("waqi_data.csv", index=False)

print(result)