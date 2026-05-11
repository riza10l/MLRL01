import csv

prices = []

with open("data_saham.csv", "r") as file:
    reader = csv.DictReader(file)
    
    for row in reader:
        prices.append(float(row["close"]))

# ambil harga terbaru
last_price = prices[0]

# hitung MA 5
ma5 = sum(prices[:5]) / 5

print("Last Price:", last_price)
print("MA5:", ma5)

# decision
if last_price > ma5:
    print("SIGNAL: BUY")
else:
    print("SIGNAL: SELL")