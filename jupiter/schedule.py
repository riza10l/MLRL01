import schedule
import time 
import subprocess

def job():
    print("update model dan chart data")
    subprocess.run(["python","call.py"])
    print("done mas")

    schedule.every().day.at("13:00").do(job)

    print("sabar masbro update dulu")
    while True:
        schedule.run_pending()
        time.sleep(1)
