import psutil
import os 
import csv
import time
from datetime import datetime
import pandas as pd
from operator import itemgetter, attrgetter
import numpy as np
import docker
import math 
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Activation, LSTM, Dropout

class Container_Hw:
        def __init__(self,  index, hw_usage):
                self.index = index
                self.hw_usage = hw_usage
        def __repr__(self):
                return repr((self.index, self.hw_usage))

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

            print(index)
            
            record = [index,  rtime, cpustatus_line[0][:-2], memstatus_line[0][:-2]] 
            
            # record the information to csv file
            container_csv = id + "_hwrecord.csv"
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

def predict_container(predict_type="both", cpu_weight=0.5, memory_weight=0.5):
    if cpu_weight + memory_weight != 1:
        raise RuntimeError("Weight Error") 
    
    # get all container id
    result_conid = get_id()

    for id in result_conid:
        id = id[:-1]

        container_csv = id + "_hwrecord.csv"
        if os.path.exists(container_csv):
            pass
        else:
            raise RuntimeError("host hardware record file not exist")
        container_info = pd.read_csv(container_csv)

        container_feature = []
        container_cpu = []
        container_memory = []
        hw_usage = []

        for i in range(len(container_info)): 
            index, time, cpu, memory = container_info.iloc[i, :]
            container_feature.append([index])
            container_cpu.append(cpu)
            container_memory.append([memory])

        # trandform to numpy array
        container_feature = np.array(container_feature, dtype=np.float64)
        container_cpu = np.array(container_cpu, dtype=np.float64)
        container_memory = np.array(container_memory, dtype=np.float64)

        if predict_type == "cpu":
            hw_usage = container_cpu
        elif predict_type == "memory":
            hw_usage = container_memory
        elif predict_type == "both":
            for i in range(len(container_cpu)):
                hw_usage.append(cpu_weight*container_cpu[i] + memory_weight*container_memory[i])

        # normalization
        ss1 = MinMaxScaler(feature_range = (0, 1))
        container_feature = ss1.fit_transform(container_feature)
        ss2 = MinMaxScaler(feature_range = (0, 1))
        hw_usage = ss2.fit_transform(hw_usage)
        print(hw_usage)

        # reshap to 3 dimension
        hw_index = np.reshape(container_feature, (container_feature.shape[0], container_feature.shape[1], 1))

        # split dataset to train dataset and test dataset
        X_train, X_test, Y_train, Y_test = train_test_split(container_feature, hw_usage, test_size=0.2)

        ### lstm method (nonlinear regression) ############################################################### 
        # adding the first LSTM layer and some Dropout regularisation
        regressor = Sequential()
        regressor.add(LSTM(units = 200, return_sequences = True, input_shape = (hw_index.shape[1], 1)))
        regressor.add(Dropout(0.2))
        # adding a second LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 200, return_sequences = True))
        regressor.add(Dropout(0.2))
        # adding a third LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 200, return_sequences = True))
        regressor.add(Dropout(0.2))
        # adding a fourth LSTM layer and some Dropout regularisation
        regressor.add(LSTM(units = 200, return_sequences = False))
        regressor.add(Dropout(0.2))
        # adding the output layer
        regressor.add(Dense(units = 1))

        # compiling
        regressor.compile(optimizer="adam", loss="mean_squared_error")

        # training
        regressor.fit(hw_index, hw_usage, epochs=2000, batch_size=32)

        del(container_feature, container_cpu, container_memory)

        # predict the next round hardwared information
        predict_index = []
        for i in range(10):
            predict_index.append([i+1])
        predict_index  = ss1.fit_transform(predict_index)
        predict_index = np.reshape(predict_index, (predict_index.shape[0], predict_index.shape[1], 1))
        predicted_hw_usage = regressor.predict(predict_index)
        # to get the original scale
        predict_index = ss1.inverse_transform(np.reshape(predict_index,(predict_index.shape[0], 1)))
        predicted_hw_usage = ss2.inverse_transform(predicted_hw_usage)
        
        # record the predict information
        container_predictcsv = id + "_predictHW.csv"
        if os.path.exists(container_predictcsv):
            csv_writer = csv.writer(open(container_predictcsv, 'a'))
            for i in range(len(predict_index)):
                csv_writer.writerow([int(predict_index[i][0]), round(predicted_hw_usage[i][0],2)])

        else:
            csv_writer = csv.writer(open(container_predictcsv, 'w'))
            csv_writer.writerow(["index", "hw_usage"])
            for i in range(len(predict_index)):
                csv_writer.writerow([int(predict_index[i][0]), round(predicted_hw_usage[i][0],2)])

        # # Visualising the results 
        # origion_hwindex = ss1.inverse_transform(np.reshape(hw_index,(hw_index.shape[0], 1)))
        # origion_hwusage = ss2.inverse_transform(np.reshape(hw_usage,(hw_usage.shape[0], 1)))
        # plt.scatter(origion_hwindex, origion_hwusage, color="red", label="previous")
        # plt.scatter(predict_index, predicted_hw_usage, color="blue", label="predict")
        # # plt.plot(Y_test, color = 'red', label = 'Real CPU Usage') 
        # # plt.plot(predicted_cpu_usage, color = 'blue', label = 'Predicted CPU Usage')  
        # plt.title('Container CPU Usage Prediction')
        # plt.xlabel('Index')
        # plt.ylabel('Usage Percent')
        # plt.legend()
        # plt.show()
        del(hw_index, hw_usage, predict_index, predicted_hw_usage)
        ### lstm method (nonlinear regression) ############################################################### 

        ### linear regression ################################################################################
        # lm = LinearRegression()
        # lm.fit(np.reshape(container_feature, (int(len(container_feature)/2), 2)), np.reshape(container_cpu, (len(container_cpu), 1)))

        # # record predict information
        # container_predcsv = id + "_predrecord.csv"
        # if os.path.exists(container_predcsv):
        #     csv_writer = csv.writer(open(container_predcsv, 'a'))
        # else:
        #     csv_writer = csv.writer(open(container_predcsv, 'w'))
        #     csv_writer.writerow(['day', 'index', 'cpu'])

        # for i in range(7):
        #     for j in range(10):
        #         to_be_predicted = np.array([i+1, j+1])
        #         predicted_cpu = lm.predict(np.reshape(to_be_predicted, (int(len(to_be_predicted)/2), 2)))
        #         predicted_record = [i+1, j+1, float(predicted_cpu)]
        #         csv_writer.writerow(predicted_record)
        #         predicted_record.clear()
        ### linear regression ################################################################################

def full_backup():
    # record information    
    fullbackup_csv = "fb_record.csv"
    if os.path.exists(fullbackup_csv):
        csv_writer = csv.writer(open(fullbackup_csv, 'a'))
    else:
        csv_writer = csv.writer(open(fullbackup_csv, 'w'))
        csv_writer.writerow(['id', 'time'])

    # get all container id
    result_conid = get_id()

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
    fb_csv = "fb_record.csv"
    fb_info = pd.read_csv(fb_csv)

    labelencoder = LabelEncoder()
    id = labelencoder.fit_transform(fb_info['id'])
    time = fb_info['time']

    fb_feature = np.array(id)
    fb_time = np.array(time)

    lm = LinearRegression()
    lm.fit(np.reshape(fb_feature, (len(fb_feature), 1)), np.reshape(fb_time, (len(fb_time), 1)))


    # record predict information
    fb_predcsv = "fb_predrecord.csv"
    if os.path.exists(fb_predcsv):
        csv_writer = csv.writer(open(fb_predcsv, 'a'))
    else:
        csv_writer = csv.writer(open(fb_predcsv, 'w'))
        csv_writer.writerow(['id', 'time'])

    for i in range(len(set(fb_feature))):
        print(i)
        to_be_predicted = np.array([i])
        predicted_cpu = lm.predict(np.reshape(to_be_predicted, (len(to_be_predicted), 1)))
        predicted_record = [i, float(predicted_cpu)]
        csv_writer.writerow(predicted_record)
        predicted_record.clear()            

                

if __name__ == "__main__":
    predict_container()