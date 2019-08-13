import psutil
import os 
import csv
import time
from datetime import datetime
import pandas as pd
from operator import itemgetter, attrgetter
import numpy as np
import math
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Activation, LSTM, Dropout



class Host_Hw:
        def __init__(self, index, hw_usage):
                self.index = index
                self.hw_usage = hw_usage
        def __repr__(self):
                return repr((self.index, self.hw_usage))

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

        print(index)

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
        info_record.append(Host_Hw(index, hw_usage))
     
    print(sorted(info_record, key=attrgetter("hw_usage")))       

def predict_host(host_csv="host_hwrecord2.csv", predict_type="memory", cpu_weight=0.5, memory_wight=0.5):
    if os.path.exists(host_csv):
        pass
    else:
        raise RuntimeError("host hardware record file not exist")
    host_info = pd.read_csv(host_csv)

    if cpu_weight + memory_weight != 1:
        raise RuntimeError("Weight Error") 

    host_feature = []
    host_cpu = []
    host_memory = []
    hw_usage = []

    for i in range(len(host_info)): 
        day, index, time, cpu, memory = host_info.iloc[i, :]
        host_feature.append([index])
        host_cpu.append([cpu])
        host_memory.append([memory])

    # trandform to numpy array
    host_feature = np.array(host_feature, dtype=np.float64)
    host_cpu = np.array(host_cpu, dtype=np.float64)
    host_memory = np.array(host_memory, dtype=np.float64)

    if predict_type == "cpu":
        hw_usage = host_cpu
    elif predict_type == "memory":
        hw_usage = host_memory
    elif predict_type == "both":
        for i in range(len(host_cpu)):
            hw_usage.append(cpu_weight*host_cpu[i] + memory_wight*host_memory[i])

    # normalization
    ss1 = MinMaxScaler(feature_range = (0, 1))
    host_feature = ss1.fit_transform(host_feature)
    ss2 = MinMaxScaler(feature_range = (0, 1))
    hw_usage = ss2.fit_transform(hw_usage)

    # reshap to 3 dimension
    hw_index = np.reshape(host_feature, (host_feature.shape[0], host_feature.shape[1], 1))

    # # split dataset to train dataset and test dataset
    # X_train, X_test, Y_train, Y_test = train_test_split(host_feature, host_cpu, test_size=0.2)

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

    del(host_feature, host_cpu, host_memory)

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
    host_predictcsv = "host_predictHW.csv"
    if os.path.exists(host_predictcsv):
        csv_writer = csv.writer(open(host_predictcsv, 'a'))
        for i in range(len(predict_index)):
            csv_writer.writerow([int(predict_index[i][0]), round(predicted_hw_usage[i][0],2)])

    else:
        csv_writer = csv.writer(open(host_predictcsv, 'w'))
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
    # plt.title('Host CPU Usage Prediction')
    # plt.xlabel('Index')
    # plt.ylabel('Usage Percent')
    # plt.legend()
    # plt.show()
    ### lstm method (nonlinear regression) ############################################################### 

    ### full connect layer method (nonlinear regression) #################################################
    # model = Sequential()
    # # print(len(set(host_feature)))
    # model.add(Dense(64, input_dim=1, init='uniform', activation='relu'))
    # model.add(Dense(32, init='uniform', activation='relu'))
    # model.add(Dense(16, init='uniform', activation='relu'))
    # model.add(Dense(8, init='uniform', activation='relu'))
    # model.add(Dense(4, init='uniform', activation='relu'))
    # model.add(Dense(1, init='uniform', activation='sigmoid'))

    # defsgd=SGD(lr=0.03)

    # model.compile(optimizer=defsgd,loss='mse')
    
    # for step in range(3000001):
    #     cost=model.train_on_batch(std_Xtrain, std_Ytrain)
    #     if step%500==0:
    #         print('cost:',cost)

    # W,b=model.layers[0].get_weights() 
    # print('W:',W,'b:',b)

    # y_pred=model.predict(std_Xtrain)

    # plt.scatter(std_Xtrain, std_Ytrain)
    # plt.plot(std_Xtrain, y_pred,'r-',lw=3)
    # plt.show()
    ### full connect layer method (nonlinear regression) #################################################

    ### linear regression ################################################################################
    #     lm = LinearRegression()
    #     lm.fit(np.reshape(host_feature, (int(len(host_feature)/2), 2)), np.reshape(host_cpu, (len(host_cpu), 1)))

    #     # record predict information
    #     host_predcsv = "host_predrecord.csv"
    #     if os.path.exists(host_predcsv):
    #         csv_writer = csv.writer(open(host_predcsv, 'a'))
    #     else:
    #         csv_writer = csv.writer(open(host_predcsv, 'w'))
    #         csv_writer.writerow(['day', 'index', 'cpu'])

    #     for i in range(7):
    #         for j in range(10):
    #             to_be_predicted = np.array([i+1, j+1])
    #             predicted_cpu = lm.predict(np.reshape(to_be_predicted, (int(len(to_be_predicted)/2), 2)))
    #             predicted_record = [i+1, j+1, float(predicted_cpu)]
    #             csv_writer.writerow(predicted_record)
    #             predicted_record.clear()  
    ### linear regression ################################################################################


if __name__ == "__main__":
    calc_space()