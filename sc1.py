import requests
import csv

API_KEY = "W1JZF7AOUMP2BIAZ"
# cari yang lain tinggal ganti symbol, sesi inter only gaada indo soalnya kikir ngentot gaada api free
symbol = "AAPL"

url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={API_KEY}"

response = requests.get(url)
data = response.json()

print(data)

# Extract time series data
time_series = data.get("Time Series (Daily)", {})

if not time_series:
    print("Error: Could not find 'Time Series (Daily)' in the response.")
    if "Note" in data:
        print(f"API Note: {data['Note']}")
    if "Error Message" in data:
        print(f"API Error: {data['Error Message']}")
    exit()

# rakit csv sendiri
with open("data_saham.csv", "w", newline="") as file:
    writer = csv.writer(file)
    
    # header    
    writer.writerow(["date", "open", "high", "low", "close", "volume"])
    
    # isi data
    for date, values in time_series.items():
        writer.writerow([
            date,
            values["1. open"],
            values["2. high"],
            values["3. low"],
            values["4. close"],
            values["5. volume"]
        ])

print("Data berhasil disimpan ke CSV!")