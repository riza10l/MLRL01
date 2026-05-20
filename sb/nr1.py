import requests
import csv

url = "https://exodus.stockbit.com/order-trade/trade-book/chart"

params = {
    "symbol": "BBCA",
    "time_interval": "1m"
}

headers = {
    "accept": "application/json, text/plain, */*",
    "authorization": "Bearer TOKEN_BARU",
    "origin": "https://stockbit.com",
    "referer": "https://stockbit.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "accept-language": "en-US,en;q=0.9,id;q=0.8",
}

res = requests.get(url, params=params, headers=headers)
data = res.json()

# cek struktur datanya dlu
print(data.keys())

# ambil bagian data "data"
rows = data["data"]

# cv ke csv
with open("bbca_1m.csv", "w", newline="") as file:
    writer = csv.writer(file)

    # header (sesuaikan kalau beda)
    writer.writerow(["timestamp", "open", "high", "low", "close", "volume"])

    for row in rows:
        writer.writerow([
            row.get("t"),
            row.get("o"),
            row.get("h"),
            row.get("l"),
            row.get("c"),
            row.get("v"),
        ])

print(" Data berhasil disimpan ke bbca_1m.csv")