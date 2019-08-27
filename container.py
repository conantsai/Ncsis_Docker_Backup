import os 
import csv
import time
from datetime import datetime
import pandas as pd
from operator import attrgetter
import numpy as np
import docker
import math 

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from predict_model import predict
from hardware_structure import Hw, Container

def get_id():
    cmd_conid = os.popen("docker ps -a -q")
    result_conid = cmd_conid.readlines()
    cmd_conid.close()
    return result_conid

def container_hwrecord(record_interval=5):
    # get all container id
    result_conid = get_id()

    # get container hardware information for all container 
    while True:
        for id in result_conid:
            id = id[:-1]

            # get the time 
            dayOfWeek = datetime.now().weekday()
            hourOfDay = int(time.strftime("%H", time.localtime()))
            minOfDay = int(time.strftime("%M", time.localtime()))
            secOfDay = int(time.strftime("%S", time.localtime()))

            # calculate the index to distribute the interval
            index = dayOfWeek*(12*24) + hourOfDay*(12) + int(minOfDay/5) + 1
            print(index)
            # get hardware information for eatch continer 
            rtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            cpustatus_cmd = "docker stats -a --no-trunc --no-stream --format  \"{{.CPUPerc}}\" " + id
            memstatus_cmd = "docker stats -a --no-trunc --no-stream --format  \"{{.MemPerc}}\" " + id
            cpustatus_output = os.popen(cpustatus_cmd)
            cpustatus_line = cpustatus_output.readlines()
            cpustatus_output.close()
            memstatus_output = os.popen(memstatus_cmd)
            memstatus_line = memstatus_output.readlines()
            memstatus_output.close()
            
            record = [index,  rtime, cpustatus_line[0][:-2], memstatus_line[0][:-2]] 
            
            # record the information to csv file
            container_csv = "./containerHW/" + id + "_hwrecord.csv"
            if os.path.exists(container_csv):
                csv_writer = csv.writer(open(container_csv, 'a'))
                csv_writer.writerow(record)
            else:
                csv_writer = csv.writer(open(container_csv, 'w'))
                csv_writer.writerow(['index','time' , 'cpu', 'memory'])
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

    return

def predict_container(predict_type="both", cpu_weight=0.5, memory_weight=0.5):
    if cpu_weight + memory_weight != 1:
        raise RuntimeError("Weight Error") 
    
    # get all container id
    result_conid = get_id()

    for id in result_conid:
        id = id[:-1]
        container_csv = "./containerHW/" + id + "_hwrecord.csv"
        predict(previous_csv=container_csv, predict_type=predict_type, cpu_weight=cpu_weight, memory_weight=memory_weight)
       
    return   

def full_backup():
    # record information    
    fullbackup_csv = "./fbrecord/fb_record.csv"
    if os.path.exists(fullbackup_csv):
        csv_writer = csv.writer(open(fullbackup_csv, 'a'))
    else:
        csv_writer = csv.writer(open(fullbackup_csv, 'w'))
        csv_writer.writerow(['id', 'time'])

    # get all container id
    result_conid = get_id()

    # execute full backup for all container and record the execute time
    for id in result_conid:
        t0 = time.time()
        id = id[:-1]

        # get the current time 
        time_stamp = int(time.time())
        time_stamp = int(time_stamp* (10 ** (10-len(str(time_stamp)))))
        time_stamp = str(time_stamp)

        client = docker.from_env()
        container = client.containers.get(id)
        conimg = container.attrs['Config']['Image']
        fb_path = "./"

        # export the container to tarfile
        container_tararchive = container.export()
        container_tar = fb_path + id + "_" + conimg + "_" + time_stamp + ".tar"
        with open(container_tar, mode = 'wb') as img_tar:
            for chunk in container_tararchive:
                img_tar.write(chunk)
        fb_exetime = time.time() - t0
        csv_writer.writerow([id, fb_exetime])
            
    return 

def predict_fb():
    # get previous information (full backup execute time)
    fb_csv = "./fbrecord/fb_record.csv"
    fb_info = pd.read_csv(fb_csv)
    id = fb_info['id']
    time = fb_info['time']

    # encode labels with value between 0 and n_classes-1. 
    labelencoder = LabelEncoder() 
    fb_feature = labelencoder.fit_transform(id)
    
    fb_feature = np.array(fb_feature)
    fb_time = np.array(time)

    # trandform to numpy array
    lm = LinearRegression()
    lm.fit(np.reshape(fb_feature, (len(fb_feature), 1)), np.reshape(fb_time, (len(fb_time), 1)))

    # record predict information (full backup execute time)
    fb_predcsv = "./fbrecord/fb_predrecord.csv"
    if os.path.exists(fb_predcsv):
        csv_writer = csv.writer(open(fb_predcsv, 'w+'))
        csv_writer.writerow(['id', 'time'])
    else:
        csv_writer = csv.writer(open(fb_predcsv, 'w'))
        csv_writer.writerow(['id', 'time'])

    for i in range(len(set(fb_feature))):
        to_be_predicted = np.array([i])
        predicted_cpu = lm.predict(np.reshape(to_be_predicted, (len(to_be_predicted), 1)))
        predicted_record = [id[i], round(float(predicted_cpu),2)]
        csv_writer.writerow(predicted_record)
        predicted_record.clear() 

    return           

def sort_backuporder():
    ## get the predict host hardware usage information and sort it
    host_csv = "./hostHW/host_predictHW.csv"
    host_info = pd.read_csv(host_csv)
    sort_host  = []
    sort_hostindex = sort_host
    sum_hwusage = 0
    
    for i in range(len(host_info)): 
        index, hw_usage = host_info.iloc[i, :]
        sort_host.append(Hw(int(index), hw_usage))
        sum_hwusage += hw_usage
    average_hwusage = sum_hwusage/len(host_info)
    sort_host = sorted(sort_host, key=attrgetter("hw_usage"))
    
    for i ,interval in enumerate(sort_host):
        sort_hostindex[i] = str(interval.index) 

    ## get the predict full backup execute time
    fullbackup_csv = "./fbrecord/fb_predrecord.csv"
    fullbackup_info = pd.read_csv(fullbackup_csv)
    
    ## get the container size information and sort it
    size_cmd = "docker ps -as --format='{{.ID}}\t{{.Size}}'"
    result_size = os.popen(size_cmd)
    size_container = result_size.readlines()
    result_size.close()

    # normalized the size unit
    sort_container=[]
    for i, info in enumerate(size_container):
        fp = info.index("virtual")
        lp = info.index(")")

        if info[lp-2:lp] == "KB":
            sort_container.append(Container(info[:12], float(info[fp+8:lp-2])*1024))
        elif info[lp-2:lp] == "MB":
            sort_container.append(Container(info[:12], float(info[fp+8:lp-2])*math.pow(1024, 2)))
        elif info[lp-2:lp] == "GB":
            sort_container.append(Container(info[:12], float(info[fp+8:lp-2])*math.pow(1024, 3)))
        elif info[lp-2:lp] == "TB":
            sort_container.append(Container(info[:12], float(info[fp+8:lp-2])*math.pow(1024, 4)))
        else:
            sort_container.append(Container(info[:12], float(info[fp+8:lp-2])))
    # sort the order of selecting interval by container size  
    sort_container = sorted(sort_container, key=attrgetter("container_size"))

    container_fbinfo = []
    container_ibinfo = []

    # get the predict hardware usage for container and sort it
    for i, container in enumerate(sort_container):
        container_csv = "./containerHW/" + container.container_id + "_predictHW.csv"
        container_info = pd.read_csv(container_csv)
        info_record = []

        for j in range(len(container_info)): 
            index, hw_usage = container_info.iloc[j, :]
            info_record.append(Hw(int(index), hw_usage))

        info_record = sorted(info_record, key=attrgetter("hw_usage"))

        # record container id and hardware usage information
        container_fbinfo.append([container.container_id,[]])
        container_ibinfo.append(container.container_id)
        for j, order in enumerate(info_record): 
            container_fbinfo[i][1].extend([order.index])

        del(info_record)

    ## get previous not backup container
    notbackup_csv = "notbackup.csv"
    if os.path.exists(notbackup_csv):
        previous_notbackup = pd.read_csv(notbackup_csv)
        for i, container in enumerate(previous_notbackup):
            container_fbinfo.remove(container)
            container_fbinfo.insert(i, container)
    else:
        pass

    ## sort the backup order
    record_all = []
    print(container_fbinfo)
    print(sort_hostindex)

    # sort the order until all container find its full backup interval
    while len(container_fbinfo) != 0:
        # according the host hardware usage
        for i, interval in enumerate(sort_hostindex):
            # according the container hardware usage for each round 
            for j in range(10):
                for k, container in enumerate(container_fbinfo): 
                    if  interval == str(container[1][j]):
                        # Calculate the required interval
                        predict_fbtime = fullbackup_info[fullbackup_info["id"].str.contains(container[0])].iloc[0][1]
                        predict_fbinterval = math.ceil(float(predict_fbtime)/60)

                        # determine whether the required interval is occupied
                        for m in range(0, predict_fbinterval):
                            if str(int(interval)+m) not in sort_hostindex: 
                                interval_can = False
                                break
                            else:
                                interval_can = True

                        # record and remove the assigned information
                        if interval_can == True:
                            record_all.append([container[0], "fb", []])
                            record_all[-1][2].extend([str(int(interval))])
                            for m in range(0, predict_fbinterval):
                                sort_hostindex.remove(str(int(interval)+m))
                                        
                            del(container_fbinfo[k])
                            break
                        elif interval_can == False:
                            continue
                else:
                    continue
                break
            else:
                continue
            break
        # all host intervals have been searched, jump out of while
        if i == len(sort_hostindex)-1:
            break
        else:
            continue

    print(container_fbinfo)
    print(record_all)
    print(sort_hostindex)

    # record not backup container
    if container_fbinfo:
        if os.path.exists(notbackup_csv):
            csv_writer = csv.writer(open( notbackup_csv, 'w+'))
            csv_writer.writerow(['id'])
            for container in container_fbinfo:
                csv_writer.writerow(container[0])
        else:
            csv_writer = csv.writer(open(notbackup_csv, 'w'))
            csv_writer.writerow(['id'])
            for container in container_fbinfo:
                csv_writer.writerow(container[0])
    
    ## sort the incremental backup if have remaining intervial
    ib_interval = []
    ib_key = 0
    # get the reamininal interval
    for i, interval in enumerate(sort_hostindex):
        for j, hwusage in enumerate(sort_host):
            if int(interval) == hwusage.index and hwusage.hw_usage < average_hwusage:
                ib_interval.append(hwusage.index)
    # assign the incremental backup
    for i, interval in enumerate(ib_interval):
        record_all.append([container_ibinfo[ib_key], "ib", [str(interval)]])
        if ib_key == len(container_ibinfo)-1:
            ib_key = 0
        else:
            ib_key += 1
        
    # record the backup order information to csv file
    backuporder_csv = "backup_order.csv"
    if os.path.exists(backuporder_csv):
        csv_writer = csv.writer(open(backuporder_csv, 'w+'))
        csv_writer.writerow(['id', 'type', 'order'])
        for i, container in enumerate(record_all):
            order_record = [container[0], container[1]]
            for j, interval in enumerate(container[2]):
                order_record.append(interval)
            csv_writer.writerow(order_record)
    else:
        csv_writer = csv.writer(open(backuporder_csv, 'w'))
        csv_writer.writerow(['id', 'type', 'order'])
        for i, container in enumerate(record_all):
            order_record = [container[0], container[1]]
            for j, interval in enumerate(container[2]):
                order_record.append(interval)
            csv_writer.writerow(order_record)

    del(sort_host, container_fbinfo, record_all, order_record)


if __name__ == "__main__":
    sort_backuporder()