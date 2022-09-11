#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import random
# normal_data_path =  {"21-02-05(1)": ["../검증 데이터/2021-02-05_정상주행1_OHT1_SegState.csv",
#                                      "../검증 데이터/2021-02-05_정상주행1_OHT2_SegState.csv",
#                                      "../검증 데이터/2021-02-05_정상주행1_OHT3_SegState.csv",
#                                      "../검증 데이터/2021-02-05_정상주행1_OHT4_SegState.csv"],
#                      "21-02-05(2)": ["../검증 데이터/2021-02-05_정상주행2_OHT1_SegState.csv",
#                                      "../검증 데이터/2021-02-05_정상주행2_OHT2_SegState.csv",
#                                      "../검증 데이터/2021-02-05_정상주행2_OHT3_SegState.csv",
#                                      "../검증 데이터/2021-02-05_정상주행2_OHT4_SegState.csv"],
#                      "21-02-05(3)": ["../검증 데이터/2021-02-05_정상주행3_OHT1_SegState.csv",
#                                      "../검증 데이터/2021-02-05_정상주행3_OHT2_SegState.csv",
#                                      "../검증 데이터/2021-02-05_정상주행3_OHT3_SegState.csv",
#                                      "../검증 데이터/2021-02-05_정상주행3_OHT4_SegState.csv"]}

# abnormal_data_path = {"Segment 이상주행": ["../검증 데이터/2021-02-07_Segment이상주행1_OHT1_SegState.csv",
#                                       "../검증 데이터/2021-02-07_Segment이상주행1_OHT2_SegState.csv",
#                                       "../검증 데이터/2021-02-07_Segment이상주행1_OHT3_SegState.csv",
#                                       "../검증 데이터/2021-02-07_Segment이상주행1_OHT4_SegState.csv",
#                                        "../검증 데이터/2021-02-07_Segment이상주행2_OHT1_SegState.csv",
#                                       "../검증 데이터/2021-02-07_Segment이상주행2_OHT2_SegState.csv",
#                                       "../검증 데이터/2021-02-07_Segment이상주행2_OHT3_SegState.csv",
#                                       "../검증 데이터/2021-02-07_Segment이상주행2_OHT4_SegState.csv"],
#                       "중부하주행": ["../검증 데이터/2021-03-07_중부하주행1(가감속 1000)_OHT1_SegState.csv",
#                                    "../검증 데이터/2021-03-07_중부하주행1(가감속 1000)_OHT2_SegState.csv",
#                                    "../검증 데이터/2021-03-07_중부하주행1(가감속 1000)_OHT3_SegState.csv",
#                                    "../검증 데이터/2021-03-07_중부하주행1(가감속 1000)_OHT4_SegState.csv"],
#                      "Manipulation" :  ["../검증 데이터/2021-08-19_정상주행3_LS-SS_OHT1_SegState.csv",
#                                        "../검증 데이터/2021-08-19_정상주행3_LC-SC_OHT1_SegState.csv",
#                                        "../검증 데이터/2021-08-19_정상주행3_LS-SC_OHT1_SegState.csv",
#                                        "../검증 데이터/2021-08-05_정상주행1_직선+곡선라벨_OHT1_SegState.csv",
#                                        "../검증 데이터/2021-08-05_정상주행2_곡선+직선라벨_OHT3_SegState.csv",
#                                        "../검증 데이터/2021-09-15_정상주행1_FOUP+NF라벨_OHT1_SegState.csv",
#                                        "../검증 데이터/2021-07-22_Hoist속도저하_OHT1_SegState.csv"]}

normal_data_path =  {"21-02-05(1)": ["../전처리 데이터 (9States)/2021-02-05_정상주행1_OHT1_9State.csv",
                                     "../전처리 데이터 (9States)/2021-02-05_정상주행1_OHT2_9State.csv",
                                     "../전처리 데이터 (9States)/2021-02-05_정상주행1_OHT3_9State.csv",
                                     "../전처리 데이터 (9States)/2021-02-05_정상주행1_OHT4_9State.csv"],
                     "21-02-05(2)": ["../전처리 데이터 (9States)/2021-02-05_정상주행2_OHT1_9State.csv",
                                     "../전처리 데이터 (9States)/2021-02-05_정상주행2_OHT2_9State.csv",
                                     "../전처리 데이터 (9States)/2021-02-05_정상주행2_OHT3_9State.csv",
                                     "../전처리 데이터 (9States)/2021-02-05_정상주행2_OHT4_9State.csv"],
                     "21-02-05(3)": ["../전처리 데이터 (9States)/2021-02-05_정상주행3_OHT1_9State.csv",
                                     "../전처리 데이터 (9States)/2021-02-05_정상주행3_OHT2_9State.csv",
                                     "../전처리 데이터 (9States)/2021-02-05_정상주행3_OHT3_9State.csv",
                                     "../전처리 데이터 (9States)/2021-02-05_정상주행3_OHT4_9State.csv"],
                    "기타" : ["../전처리 데이터 (9States)/2021-08-24_정상주행3-Hoist_OHT1_9State.csv",
                              "../전처리 데이터 (9States)/2021-08-05_정상주행1-STB_OHT1_9State.csv"]}
abnormal_data_path = {
#     "Segment 이상주행": ["../전처리 데이터 (9States)/2021-02-07_Segment이상주행1_OHT1_9State.csv",
#                                       "../전처리 데이터 (9States)/2021-02-07_Segment이상주행1_OHT2_9State.csv",
#                                       "../전처리 데이터 (9States)/2021-02-07_Segment이상주행1_OHT3_9State.csv",
#                                       "../전처리 데이터 (9States)/2021-02-07_Segment이상주행1_OHT4_9State.csv",
#                                        "../전처리 데이터 (9States)/2021-02-07_Segment이상주행2_OHT1_9State.csv",
#                                       "../전처리 데이터 (9States)/2021-02-07_Segment이상주행2_OHT2_9State.csv",
#                                       "../전처리 데이터 (9States)/2021-02-07_Segment이상주행2_OHT3_9State.csv",
#                                       "../전처리 데이터 (9States)/2021-02-07_Segment이상주행2_OHT4_9State.csv"],
                      "중부하주행": ["../전처리 데이터 (9States)/2021-03-07_중부하주행1(가감속 1000)_OHT1_9State.csv",
                                   "../전처리 데이터 (9States)/2021-03-07_중부하주행1(가감속 1000)_OHT2_9State.csv",
                                   "../전처리 데이터 (9States)/2021-03-07_중부하주행1(가감속 1000)_OHT3_9State.csv",
                                   "../전처리 데이터 (9States)/2021-03-07_중부하주행1(가감속 1000)_OHT4_9State.csv"],
                      
                          "기타" : [ "../전처리 데이터 (9States)/2021-03-11_트랙이상(종이,500)_OHT1_9State.csv",
                                  "../전처리 데이터 (9States)/2021-03-11_트랙이상refine_OHT1_9State.csv",
                                   "../전처리 데이터 (9States)/2021-07-22_Hoist속도변경_OHT1_9State.csv",
                                  "../전처리 데이터 (9States)/2021-07-22_Hoist속도저하_OHT1_9State.csv",
                                  "../전처리 데이터 (9States)/2021-08-24_Hoist토크증가_OHT1_9State.csv",
                                    "../전처리 데이터 (9States)/2021-02-05_Hoist모터과부하(2)_OHT4_9State.csv",
                                 "../전처리 데이터 (9States)/2021-02-05_Hoist모터과부하(2)-refine_OHT4_9State.csv"],
                      
                      "Manipulation" : ["../DataManipulation/2021-08-05_정상주행1_직선+곡선라벨_OHT1_9State.csv",
                                        "../DataManipulation/2021-08-05_정상주행2_곡선+직선라벨_OHT3_9State.csv",
                                       "../DataManipulation/2021-08-05_정상주행1_STB+Station라벨_OHT1_9State.csv",
                                        "../DataManipulation/2021-08-05_정상주행1_STB+직선라벨_OHT1_9State.csv",
                                         "../DataManipulation/2021-04-19_정상주행1_STB+Station_OHT1_9State.csv",
                                       "../DataManipulation/2021-04-19_정상주행1_STB+Station+Station라벨_OHT1_9State.csv",
                                       "../DataManipulation/2021-04-19_정상주행2_주행+LUL_OHT2_9State.csv",
                                       "../DataManipulation/2021-04-19_정상주행2_주행+LUL+STB라벨_OHT2_9State.csv",
                                       "../DataManipulation/2021-09-15_정상주행1_FOUP+NF라벨_OHT1_9State.csv",
                                       "../DataManipulation/2021-08-24_정상주행3-Hoist+직선라벨_OHT1_9State.csv"]}




sensors = ["X_RMS","Y_RMS","Z_RMS","speed1","torque1", "speed2","torque2", "speed3","torque3","speed4","torque4", "roll"]
batch_size = time_window = sliding_step = train_min = train_max = 0
n_features = len(sensors)
n_label = 9 #0~8

def normalizer(df):
    normalized_df =(df-df.min())/(df.max()-df.min())
    return normalized_df

def test_normalizer(df, train_min, train_max):
    normalized_df = (df-train_min)/(train_max-train_min)
    return normalized_df

def one_hot(label):
    target = np.zeros(n_label)
    target[label] = 1
    return target.astype("float32")

    
def get_sw_data(data, df, time_window, sliding_step, model='FCAE', phase = "test", speeding = False):
    label_df = df["State_Event"]
    new_data = []
    for i in range(0,len(data)-time_window,sliding_step):
        if data.index[i+time_window] == data.index[i] + time_window:
            if model in ["FCAE", "VAE","CVAE"]:
                sensor = np.array(data.iloc[i:i+time_window]).T.reshape(-1).astype("float32") #(1200,1)
            elif "RAE" in model:
                sensor = np.array(data.iloc[i:i+time_window]).astype("float32") #(100,12)
            elif model == "CAE":
                sensor = np.array(data.iloc[i:i+time_window]).T.astype("float32") 
                sensor = np.expand_dims(sensor,-1)                                 #(12, 100, 1)
            else:
                raise NotImplementedError
            
            if phase == "train":
                if np.max(sensor) > 1 or np.min(sensor) < -1:
                    continue
            if speeding:
                if np.max(np.array(data["speed1"].iloc[i:i+time_window])) < 1:
                    continue
                
            sw_label = list(label_df.iloc[i:i+time_window])
            label = max(set(sw_label), key = sw_label.count)

            if sw_label.count(label) >= 0.95 * len(sw_label): #noise 5개까지 허용
                if label != 0:
                    new_data.append((sensor, (one_hot(label), data.index[i])))
    return new_data


def get_sw_data_node(data, df, time_window, sliding_step, model='FCAE', phase="test", speeding = False):    
    print(speeding)
    for i in range(len(df)):
        if df["State_Segment"].iloc[i] in [1,2,3] and df["State_Event"].iloc[i] in [2,6]: #직선->곡선 gap 보정
            if df["roll"].iloc[i] > 0:
                df["State_Segment"].iloc[i] =4
            else:
                df["State_Segment"].iloc[i] = 5
        if df["State_Segment"].iloc[i] in [4,5] and df["State_Event"].iloc[i] in [1,5]: # 곡선->직선 gap 보정
            df["State_Segment"].iloc[i] =1
             
        if df["State_Event"].iloc[i] >= 5: #FOUP 유무 추가
            df["State_Segment"].iloc[i] = df["State_Segment"].iloc[i]*2     

    label_df = df["State_Event"]
    VCS_df = df["State_Segment"]

    LastNode = df["LastNode"]
    NextNode = df["NextNode"]

    new_data = []
    for i in range(0, len(data) - time_window, sliding_step):
        if data.index[i + time_window] == data.index[i] + time_window:
            if model in ["FCAE", "VAE", "CVAE", "CondAE"]:
                sensor = np.array(data.iloc[i:i + time_window]).T.reshape(-1).astype("float32")  # (1200,1)
            elif "RAE" in model:
                sensor = np.array(data.iloc[i:i + time_window]).astype("float32")  # (100,12)
            elif model == "CAE":
                sensor = np.array(data.iloc[i:i + time_window]).T.astype("float32")
                sensor = np.expand_dims(sensor, -1)  # (12, 100, 1)
            else:
                raise NotImplementedError

            if phase == "train":
                if np.max(sensor) > 1 or np.min(sensor) < -1:
                    continue
            if speeding:
                if np.max(np.array(data["speed1"].iloc[i:i+time_window])) < 1:
                    continue

            sw_label = list(label_df.iloc[i:i + time_window])
            label = max(set(sw_label), key=sw_label.count)

            sw_label_VCS = list(VCS_df.iloc[i:i + time_window])
            label_VCS = max(set(sw_label_VCS), key=sw_label_VCS.count)
            
            if sw_label_VCS.count(label_VCS) == len(sw_label_VCS):
                if label != 0 and label_VCS != 0:
                    new_data.append((sensor, (one_hot(label_VCS), data.index[i])))
    return new_data


def prepare_train_data(df, time_window, sliding_step, model):
    df["speed3"] = -df["speed3"]
    df["torque3"] = -df["torque3"]
    #df["torque2"] = -df["torque2"]
    df["roll"] = -df["roll"]
    
    data = df[sensors]
    label = df["State_Event"]
    #label_VCS = df["State_VCS"]
    #label_Seg = df["State_Segment"]
    
#     label[label==5] = 1
#     label[label==6] = 2
#     label[label==7] = 3
#     label[label==8] = 4
    
    train_min = np.sort(df[sensors].values, axis = 0)[10]
    train_max = np.sort(df[sensors].values, axis = 0)[::-1][10]
    train_min = pd.Series(train_min, index = sensors)
    train_max = pd.Series(train_max, index = sensors)
    
    data = normalizer(data)
    data = get_sw_data(data, df, time_window, sliding_step, model, "train")
    random.shuffle(data)

    return data, train_min, train_max


def prepare_test_data(df, batch_size, train_min, train_max, time_window, sliding_step, model, speeding=False):
    print(speeding)
    df["speed3"] = -df["speed3"]
    df["torque3"] = -df["torque3"]
    #df["torque2"] = -df["torque2"]
    df["roll"] = -df["roll"]
    
    
    data = df[sensors]
    label = df["State_Event"]
    #label_VCS = df["State_VCS"]
    #label_Seg = df["State_Segment"]
    
#     label[label==5] = 1
#     label[label==6] = 2
#     label[label==7] = 3
#     label[label==8] = 4

    data = test_normalizer(data, train_min, train_max)
    data = get_sw_data(data, df, time_window, sliding_step, model, speeding=speeding)

    return data


def load_train_data(train_date, batch_size, time_window, sliding_step, model):
    global train_min, train_max

    train_df_list = []

    for i, path in enumerate(normal_data_path[train_date]):
        print(path)
        train_df = pd.read_csv(path, index_col="Unnamed: 0")[:100000]
        train_df_list.append(train_df)
    train_total_df = pd.concat(train_df_list)
    train_total, train_min, train_max = prepare_train_data(train_total_df, time_window, sliding_step, model)
    print("<train_min>\n", train_min, "\n", "<train_max>\n", train_max)

    random.shuffle(train_total)
    train_len = int(len(train_total) * 0.7)
    val_len = len(train_total) - train_len
    test_len = len(train_total) - train_len - val_len
    train = train_total[:train_len]
    val = train_total[train_len:]

    dataloaders = {}
    dataloaders['train'] = DataLoader(train, batch_size=batch_size, shuffle=True)
    dataloaders['val'] = DataLoader(val, batch_size=batch_size, shuffle=False)

    return dataloaders


def load_test_data(train_date, batch_size, time_window, sliding_step, model):
    test_normal = {}
    test_abnormal = {}

    for path in normal_data_path[train_date]:
        print(path)
        test_df = pd.read_csv(path, index_col="Unnamed: 0")[100000:]
        test_data = prepare_test_data(test_df, batch_size, train_min, train_max, time_window, sliding_step, model)
        test_normal[path] = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for date in normal_data_path:
        if date != train_date:
            for path in normal_data_path[date]:
                print(path)
                test_df = pd.read_csv(path, index_col="Unnamed: 0")
                test_data = prepare_test_data(test_df, batch_size, train_min, train_max, time_window, sliding_step,
                                              model)
                test_normal[path] = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    for date in abnormal_data_path:
        for path in abnormal_data_path[date]:
            print(path)
            test_df = pd.read_csv(path, index_col="Unnamed: 0")
            if "중부하" in path:
                test_data = prepare_test_data(test_df, batch_size, train_min, train_max, time_window, sliding_step, model, speeding=True)
            else:
                test_data = prepare_test_data(test_df, batch_size, train_min, train_max, time_window, sliding_step, model)
            test_abnormal[path] = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return test_normal, test_abnormal