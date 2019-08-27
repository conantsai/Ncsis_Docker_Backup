import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Activation, LSTM, Dropout

def predict(previous_csv, predict_type, cpu_weight, memory_weight):
    if os.path.exists(previous_csv):
        pass
    else:
        raise RuntimeError("hardware record file not exist")
    previous_info = pd.read_csv(previous_csv)

    previous_feature = []
    previous_cpu = []
    previous_memory = []
    hw_usage = []

    for i in range(len(previous_info)): 
        index, time, cpu, memory = previous_info.iloc[i, :]
        previous_feature.append([index])
        previous_cpu.append(cpu)
        previous_memory.append([memory])

    # trandform to numpy array
    previous_feature = np.array(previous_feature, dtype=np.float64)
    previous_cpu = np.array(previous_cpu, dtype=np.float64)
    previous_memory = np.array(previous_memory, dtype=np.float64)

    if predict_type == "cpu":
        hw_usage = previous_cpu
    elif predict_type == "memory":
        hw_usage = previous_memory
    elif predict_type == "both":
        for i in range(len(previous_cpu)):
            hw_usage.append(cpu_weight*previous_cpu[i] + memory_weight*previous_memory[i])

    # normalization
    ss1 = MinMaxScaler(feature_range = (0, 1))
    previous_feature = ss1.fit_transform(previous_feature)
    ss2 = MinMaxScaler(feature_range = (0, 1))
    hw_usage = ss2.fit_transform(hw_usage)

    # reshap to 3 dimension
    hw_index = np.reshape(previous_feature, (previous_feature.shape[0], previous_feature.shape[1], 1))

    # # split dataset to train dataset and test dataset
    # X_train, X_test, Y_train, Y_test = train_test_split(previous_feature, hw_usage, test_size=0.2)

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

    # predict the next round hardwared information
    predict_indexo = []
    for i in range(10):
        predict_indexo.append([i+1])

    predict_index = ss1.fit_transform(predict_indexo)
    predict_index = np.reshape(predict_index, (predict_index.shape[0], predict_index.shape[1], 1))
    predicted_hw_usage = regressor.predict(predict_index)
    # to get the original scale
    predict_index = ss1.inverse_transform(np.reshape(predict_index,(predict_index.shape[0], 1)))
    predicted_hw_usage = ss2.inverse_transform(predicted_hw_usage)
    
    # record the predict information
    underline_pos = previous_csv.index("_")
    predictcsv = previous_csv[:underline_pos] + "_predictHW.csv"
    if os.path.exists(predictcsv):
        csv_writer = csv.writer(open(predictcsv, 'w+'))
        csv_writer.writerow(["index", "hw_usage"])
        for i in range(len(predict_index)):
            csv_writer.writerow([int(predict_indexo[i][0]), round(predicted_hw_usage[i][0],2)])
    else:
        csv_writer = csv.writer(open(predictcsv, 'w'))
        csv_writer.writerow(["index", "hw_usage"])
        for i in range(len(predict_index)):
            csv_writer.writerow([int(predict_indexo[i][0]), round(predicted_hw_usage[i][0],2)])

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

    del(previous_feature, previous_cpu, previous_memory, hw_index, hw_usage, predict_index, predicted_hw_usage)
    ## lstm method (nonlinear regression) ############################################################### 

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
    #     cost=model.train_on_batch(hw_index, hw_usage)
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
    #     lm.fit(np.reshape(previous_feature, (int(len(previous_feature)/2), 2)), np.reshape(hw_usage, (len(hw_usage), 1)))

    #     # record predict information
    #     underline_pos = previous_csv.index("_")
    #     container_predictcsv = previous_csv[:underline_pos] + "_predictHW.csv"
    #     if os.path.exists(container_predictcsv):
    #         csv_writer = csv.writer(open(container_predictcsv, 'a'))
    #     else:
    #         csv_writer = csv.writer(open(container_predictcsv, 'w'))
    #         csv_writer.writerow(["index", "hw_usage"])

    #     for i in range(7):
    #         for j in range(10):
    #             to_be_predicted = np.array([i+1, j+1])
    #             predicted_cpu = lm.predict(np.reshape(to_be_predicted, (int(len(to_be_predicted)/2), 2)))
    #             predicted_record = [i+1, j+1, float(predicted_cpu)]
    #             csv_writer.writerow(predicted_record)
    #             predicted_record.clear()  
    ### linear regression ################################################################################

    return


if __name__ == "__main__":
    # test host predict
    predict(previous_csv="host_hwrecord2.csv", predict_type="both", cpu_weight=0.5, memory_weight=0.5)
    
    # test container predict
    cmd_conid = os.popen("docker ps -a -q")
    result_conid = cmd_conid.readlines()
    cmd_conid.close()

    for id in result_conid:
        id = id[:-1]

        container_csv = id + "_hwrecord.csv"
        predict(previous_csv=container_csv, predict_type="both", cpu_weight=0.5, memory_weight=0.5)