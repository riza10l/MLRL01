import yfinance as yf
import pandas as pd
import glob
import re
import os
import platform
from datetime import date, timedelta

symbol = "GC=F"
end_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")

print(f"📥 Download data {symbol} sampai {end_date}...")
data = yf.download(symbol, start="2010-01-01", end=end_date, interval="1d")

print("load data sek yo mas")

if not data.empty:
    data.reset_index(inplace=True)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
    data.columns = [col.lower() for col in data.columns]

    print("jumlah data:", len(data))
    print(data.head())
    #date time dll auto increment 
    existing_files = glob.glob("data*.csv")
    max_num = 0
    for f in existing_files:
        match = re.search(r'data(\d+)\.csv', f)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num

    next_num = max_num + 1
    filename = f"data{next_num:02d}.csv"  # padding 2 digit: 01, 02, 03...

    data.to_csv(filename, index=False)
    print(f" Data berhasil disimpan ke: {filename}")

    print(" Membuka file...")
    abs_path = os.path.abspath(filename)
    if platform.system() == "Windows":
        os.startfile(abs_path)
    elif platform.system() == "Darwin":
        os.system(f"open '{abs_path}'")
    else:  # Linux
        os.system(f"xdg-open '{abs_path}'")
else:
    print('goblok ni tolol')