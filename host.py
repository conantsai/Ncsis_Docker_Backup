import psutil
import os 
import csv
import time
from datetime import datetime
import pandas as pd
from operator import attrgetter
import math

from predict_model import predict
from hardware_structure import Hw

def host_hwrecord(record_interval=5):
    # open the csv file to record information
    host_csv = "host_hwrecord11.csv"
    if os.path.exists(host_csv):
        csv_writer = csv.writer(open(host_csv, 'a'))
    else:
        csv_writer = csv.writer(open(host_csv, 'w'))
        csv_writer.writerow(['index','time', 'cpu', 'memory'])

    while True:
        # get the time 
        dayOfWeek = datetime.now().weekday()
        hourOfDay = int(time.strftime("%H", time.localtime()))
        minOfDay = int(time.strftime("%M", time.localtime()))
        secOfDay = int(time.strftime("%S", time.localtime()))

        # calculate the index to distribute the interval
        index = dayOfWeek*(12*24) + hourOfDay*(12) + int(minOfDay/5) + 1

        # get host hardware information 
        rtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
        memory = psutil.virtual_memory().percent
        cpu = psutil.cpu_percent(interval=1)
        record = [index, rtime, cpu, memory]
        
        # record the information to csv file
        csv_writer.writerow(record)
        record.clear()

        # calculate sleep time to start next round to record
        if (minOfDay%record_interval) == 0:
            sleep_min = record_interval - 1 
        else:
            sleep_min = math.ceil(minOfDay/record_interval)*record_interval - minOfDay - 1
        if sleep_min < 0:
            sleep_time = 60 - secOfDay
        elif sleep_min > 0: 
            sleep_time = sleep_min*60 + (60 - secOfDay)
        elif sleep_min == 0:
            sleep_time = (record_interval+1) + (60 - secOfDay)
        
        time.sleep(sleep_time)
    
def calc_space():
    host_csv = "host_predictHW.csv"
    host_info = pd.read_csv(host_csv)
    info_record = []
    
    for i in range(len(host_info)): 
        index, hw_usage = host_info.iloc[i, :]
        info_record.append(Hw(index, hw_usage))
     
    print(sorted(info_record, key=attrgetter("hw_usage")))       

def predict_host(host_csv="./hostHW/host_hwrecord.csv", predict_type="both", cpu_weight=0.5, memory_weight=0.5):
    predict(previous_csv=host_csv, predict_type=predict_type, cpu_weight=cpu_weight, memory_weight=memory_weight)

if __name__ == "__main__":
    predict_host()