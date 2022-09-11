#!/usr/bin/env python
# coding: utf-8

import time
import copy
import numpy as np
import pandas as pd
import train
from utils import load_test_data

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import metrics


args = train.args
device = train.device
loss_func = args.loss_func
n_features = args.n_features
time_window = args.time_window
sensors = ["X_RMS","Y_RMS","Z_RMS","speed1","torque1", "speed2","torque2", "speed3","torque3","speed4","torque4", "roll"]

state_labels = {0:"stop", 1:"str+noFOUP", 2:"crv+noFOUP", 
                          3:"STB+noFOUP", 4:"Stn+noFOUP", 
                          5:"str+FOUP", 6:"crv+FOUP", 
                          7:"STB+FOUP", 8:"Stn+FOUP"}

# state_labels = {0:"Stop", 1:"Line", 2:"Short Arc", 3:"Long Arc", 4:"STB", 5:"Station"}
state_labels = {0:"Stop", 1:"Line", 2:"Arc", 3:"STB", 4:"Station"}
#state_labels = {0:"Stop", 1:"Short Line", 2:"Mid Line", 3:"Long Line", 4:"Short Curve", 5:"Long Curve", 6:"STB", 7:"Station"} 
# state_labels = {0:"Stop", 1:"Short Line", 2:"Mid Line", 3:"Long Line",
#                 4:"Right Curve", 5:"Left Curve", 6:"STB", 7:"Station",
#                 8:"Short Line + FOUP", 9:"Mid Line + FOUP", 10:"Long Line + FOUP", 
#                 11:"Right Curve + FOUP", 12: "Left Curve + FOUP", 13:"STB+FOUP", 14:"Station+FOUP"} 

def get_test_output(test_data, best_model, model_type= "FCAE"):
    outputs_list = []
    loss_list = []
    z_list = []
    target_list = []
    with torch.no_grad():
        running_loss = 0.0
        for (inputs, (labels, index)) in test_data:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if model_type in ["VAE","CVAE"]:
                outputs, mu, log_var, latent = best_model(inputs, labels)
            else:
                outputs, latent = best_model(inputs, labels)
            
            for i in range((len(inputs))):
                if model_type == "ARAE":
                    c_repeat = torch.stack([labels[i] for _ in range(args.time_window)], dim = 0) #(100,9)
                    concat_inputs = torch.cat((inputs[i], c_repeat), dim=1) #(100,21)
                    test_loss = args.loss_func(concat_inputs, outputs[i])
                    loss_list.append(test_loss.cpu())
                    outputs_list.append(outputs[i].cpu().numpy())
                    z_list.append(latent[i].cpu().numpy())
                    target_list.append(labels[i].cpu().tolist().index(1))
                else:
                    test_loss = loss_func(outputs[i], inputs[i])
                    loss_list.append(test_loss.cpu())
                    outputs_list.append(outputs[i].cpu().numpy())
                    z_list.append(latent[i].cpu().numpy())
                    target_list.append(labels[i].cpu().tolist().index(1))


        test_loss = np.array(loss_list).mean()
        print("Loss: {:5f}".format(test_loss)) 

    return np.array(outputs_list), np.array(loss_list), np.array(z_list), target_list


def plot_roc_curve(test_normal, test_abnormal, test_loss_dict, model_type = "FCAE", target = "Segment"):
    loss_score = []
    true_label = []
    normal_score = []
    abnormal_score = []
    for date in test_loss_dict["N"]:
        if "STB" in target:
            if "STB" not in date:
                continue
        for i in range(0,len(test_loss_dict["N"][date]),1):
            state = np.where(test_normal[date].dataset[i][1][0]==1)[0].item()
            if target in ["Segment", "중부하"]:
                if state in [1,5]:
                    loss = np.array(sorted(test_loss_dict["N"][date][i:i+300], reverse = True)[0:5]).mean()
                    #loss = test_loss_dict["N"][date][i]
                else: continue
            elif "직선" in target:
                if state in [1,5] :
                    loss = test_loss_dict["N"][date][i]
                else: continue
            elif "곡선" in target:
                if state in [2,6] :
                    loss = test_loss_dict["N"][date][i]
                else: continue
            elif "트랙이상" in target:
                if state in [1,5] :
                    loss = test_loss_dict["N"][date][i]
                else: continue
            elif "Hoist" in target or "Torque" in target:
                if state in [4,8] :
                    loss = test_loss_dict["N"][date][i]
                else: continue
            elif "STB" in target:
                if state in [3,7] :
                    loss = test_loss_dict["N"][date][i]
                else: continue
            elif "FOUP" in target:
                if state in [8]:
                    loss = test_loss_dict["N"][date][i]
                else: continue
            else:
                loss = test_loss_dict["N"][date][i]
            loss_score.append(loss)
            normal_score.append(loss)
            true_label.append(0)
    for date in test_loss_dict["ABN"]:
        if target in date:
            print(date)
            for i in range(0,len(test_loss_dict["ABN"][date]),1):
                state = np.where(test_abnormal[date].dataset[i][1][0]==1)[0].item()
                if target in ["Segment", "중부하"]:
                    if state in [1,5]:
                        loss = np.array(sorted(test_loss_dict["ABN"][date][i:i+300], reverse = True)[0:5]).mean()
                        #loss = test_loss_dict["ABN"][date][i]
                    else: continue
                else:
                    loss = test_loss_dict["ABN"][date][i]
                loss_score.append(loss)
                abnormal_score.append(loss)
                true_label.append(1)
    
    print(np.mean(normal_score), np.mean(abnormal_score))
    print(target)
    auc = metrics.roc_auc_score(true_label, loss_score)
    pr = metrics.average_precision_score(true_label, loss_score)
    print("AUROC: ", auc)
    print("AUPR: ", pr)
    
    fpr, tpr, thresholds = metrics.roc_curve(true_label, loss_score, pos_label=1)
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate", fontsize = 15)
    plt.ylabel("True Positive Rate", fontsize = 15)
    plt.ylim(-0.05,1.05)
    plt.title("ROC Curve ({})".format(model_type), fontsize =20, pad=15)
    plt.text(0.7,0.2,"AUC={:.4f}".format(auc),fontsize=15)
    plt.show()
    
    
    precision, recall, thresholds = metrics.precision_recall_curve(true_label, loss_score)
    plt.plot(recall, precision)
    plt.xlabel("Recall", fontsize = 15)
    plt.ylabel("Precision", fontsize = 15)
    plt.ylim(-0.05,1.05)
    plt.title("P-R Curve ({})".format(model_type), fontsize =20, pad=15)
    plt.text(0.1,0.5,"AUPR={:.4f}".format(pr),fontsize=15)
    plt.show()
    return normal_score, abnormal_score

def plot_roc_curve_segment(test_normal, test_abnormal, test_loss_dict, model_type = "FCAE", target = "Segment"):
    loss_score = []
    true_label = []
    normal_score = []
    abnormal_score = []
    for date in test_loss_dict["N"]:
        for i in range(0,len(test_loss_dict["N"][date]),1):
            state = np.where(test_normal[date].dataset[i][1][0]==1)[0].item()
            if target in ["Segment", "중부하"]:
                if state in [3]:
                    loss = np.array(sorted(test_loss_dict["ABN"][date][i:i+300], reverse = True)[0:5]).mean()
                    #loss = test_loss_dict["N"][date][i]
                else: continue
            elif "직선" in target:
                if state in [1,2,3] :
                    loss = test_loss_dict["N"][date][i]
                else: continue
            elif "곡선" in target:
                if state in [4,5] :
                    loss = test_loss_dict["N"][date][i]
                else: continue
            elif "Hoist" in target:
                if state in [7] :
                    loss = test_loss_dict["N"][date][i]
                else: continue
            else:
                loss = test_loss_dict["N"][date][i]
            loss_score.append(loss)
            normal_score.append(loss)
            true_label.append(0)
    for date in test_loss_dict["ABN"]:
        if target in date:
            print(date)
            for i in range(0,len(test_loss_dict["ABN"][date]),1):
                state = np.where(test_abnormal[date].dataset[i][1][0]==1)[0].item()
                if target in ["Segment", "중부하"]:
                    if state in [3]:
                        loss = np.array(sorted(test_loss_dict["ABN"][date][i:i+300], reverse = True)[0:5]).mean()
                        #loss = test_loss_dict["ABN"][date][i]
                    else: continue
                else:
                    loss = test_loss_dict["ABN"][date][i]
                loss_score.append(loss)
                abnormal_score.append(loss)
                true_label.append(1)
    
    print(np.mean(normal_score), np.mean(abnormal_score))
    print(target)
    auc = metrics.roc_auc_score(true_label, loss_score)
    fpr, tpr, thresholds = metrics.roc_curve(true_label, loss_score, pos_label=1)
    plt.plot(fpr,tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("{}".format(model_type), fontsize =18)
    plt.text(0.7,0.2,"AUROC={:.4f}".format(auc),fontsize=15)
    plt.show()
    return normal_score, abnormal_score

#Loss Histogram

def loss_histogram(normal_score, abnormal_score, model_type):
    plt.hist(normal_score, bins = 100, label = "Normal")
    plt.xlabel("Loss")
    plt.title("Loss Histogram")
    plt.legend()
    plt.show()

    plt.hist(abnormal_score,  bins = 100, color = "orange",label = "Abnormal")
    plt.xlabel("Loss")
    plt.title("Loss Histogram")
    plt.legend()
    plt.show()

    plt.hist(normal_score, bins = 100,range = (0,max(abnormal_score)),  label = "Normal")
    plt.hist(abnormal_score, alpha = 0.5,  bins = 100, range = (0,max(abnormal_score)), color = "orange", label = "Abnormal")

    plt.xlabel("Loss", fontsize =15)
    plt.title("Loss Histogram ({})".format(model_type), fontsize =20, pad=15)
    #plt.xlim([-0.005, 0.025])
    plt.legend()
    plt.show()
    
    
#Loss by States
def loss_by_state(path_list, test_normal, test_abnormal, test_loss_dict):
    for path in path_list:
        loss_state = {}
        for s in state_labels.keys():
            loss_state[s] = [0.0]
        if path in test_normal.keys():
            print(path.split("/")[-1])
            for (i, data) in enumerate(test_normal[path].dataset):
                label = list(data[1][0]).index(1)
#                 if label not in [4,8]:
#                     label = label%4
                loss_state[label].append(test_loss_dict["N"][path][i])
            for (k,v) in loss_state.items():
                print("{}:{:.3g}".format(state_labels[k], np.mean(v)), end=" ")
            print("")
            print("")
        elif path in test_abnormal.keys():
            print(path.split("/")[-1])
            for (i, data) in enumerate(test_abnormal[path].dataset):
                label = list(data[1][0]).index(1)
#                 if label in [4,8]:
#                     label = 4
#                 else:
#                     label = label%4
                loss_state[label].append(test_loss_dict["ABN"][path][i])
            for (k,v) in loss_state.items():
                print("{}:{:.3g}".format(state_labels[k], np.mean(v)), end=" ")  
            print("")
            print("")
        else:
            raise NotImplementedError
        




# Reconstruction Plot

def plot_sample(test_normal, test_abnormal, test_out_dict, test_loss_dict, date, idx, data_class="ABN", model_type = "FCAE", org_dataloader = None):
    sensors_name = {"X_RMS":"X RMS","Y_RMS":"Y RMS","Z_RMS":"Z RMS",
                    "speed1":"Front SP","torque1":"Front TQ", "speed2":"Hoist SP","torque2":"Hoist TQ", 
                    "speed3":"Rear SP","torque3":"Rear TQ","speed4":"Slide SP","torque4":"Slide TQ", "roll":"Yaw"}
    sensors_idx = {"X_RMS":(0,0),"Y_RMS":(0,1),"Z_RMS":(0,2), "roll":(0,3),
                   "speed1" :(1,0),"torque1":(1,1), "speed2":(0,4),"torque2":(0,5), 
                   "speed3":(1,2),"torque3":(1,3),"speed4":(1,4),"torque4":(1,5)}
    name = date.split("/")[-1]
    print(name)
    if data_class == "N":
        data = test_normal[date]
    else:
        data = test_abnormal[date]
        
    if "RAE" in model_type: 
        input_all = data.dataset[idx][0].T.reshape(-1)
        out_all = np.array(test_out_dict[data_class][date][idx].T.reshape(-1))
        if org_dataloader:
            org = org_dataloader.dataset[idx][0].T.reshape(-1)
    else:
        input_all = data.dataset[idx][0].reshape(-1)
        out_all = np.array(test_out_dict[data_class][date][idx].reshape(-1))
        if org_dataloader:
            org = org_dataloader.dataset[idx][0].reshape(-1)
    
    fig, axes = plt.subplots(2,6,figsize=(10,5))
    for feature in range(n_features):
        if feature < 3:
            row = 0 
            col = feature
            color = "darkorange"
        elif feature == 11:
            row = 0
            col = 3
            color = "forestgreen"
        else:
            row = (feature +1)//4
            col = (feature +1)%4
            color = "#1f77b4"
        ax = axes[sensors_idx[sensors[feature]]]
        input_sample = input_all.reshape(n_features, time_window).T[:,feature]
        out_sample = out_all.reshape(n_features, time_window).T[:,feature]
        
        if org_dataloader:
            org_sample = org.reshape(n_features, time_window).T[:,feature]
            ax.plot(input_sample, label = "Manipulated Input", color = "red")
            ax.plot(org_sample, label = "Original Input")
            ax.plot(out_sample, label = "Reconstruction")
        else:
            ax.plot(input_sample, label = "Input")
            ax.plot(out_sample, label = "Reconstruction")
        #ax.text(0.5,0.8, "loss : {:.2g}".format(np.square(input_sample-out_sample).mean()), fontsize=13,transform=ax.transAxes)
        loss = np.square(input_sample-out_sample).mean()
        ax.set_xlabel("loss: {:.2g}".format(loss), fontsize=12)
        ax.set_title(sensors_name[sensors[feature]], fontsize = 15)
        ax.set_ylim(min(0,min(np.r_[input_sample, out_sample])-0.1),max(max(np.r_[input_sample, out_sample])+0.1,1))
        #ax.set_ylim(-0.5,2.5)
        
    label = list(data.dataset[idx][1][0]).index(1)
#     LastNode = data.dataset[idx][1][2]
#     NextNode = data.dataset[idx][1][3]
        
    #fig.subplots_adjust(top=5, left=0.1, right=0.9, bottom=0.12)
    handles, labels = axes[1][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.tight_layout(pad=0.5)
    #fig.text(0.5,1.1,'{} (loss : {:.2g})'.format(model_type, np.square(input_all-out_all).mean()),ha='center',va ='top', fontsize = 30)
    fig.text(0.5,1.1,'{} (loss : {:.2g}), {}'.format(model_type, np.square(input_all-out_all).mean(),
                                                              state_labels[label]),ha='center',va ='top', fontsize = 20)
    #fig.text(0.5,1.1,"Manipulated Data Sample",ha='center',va ='top', fontsize = 30)
    
    plt.show()



def plot_sample_all(test_normal, test_abnormal,test_out_dict, test_loss_dict, date, idx, data_class = "ABN", model_type = "FCAE",org_dataloader = None):
    name = date.split("/")[-1]
    print(name)
    fig, ax = plt.subplots(figsize=(14,6))
    if data_class == "N":
        data = test_normal[date]
    else:
        data = test_abnormal[date]
              
    if "RAE" in model_type:
        input_sample = data.dataset[idx][0].T.reshape(-1)
        out_sample = np.array(test_out_dict[data_class][date][idx].T.reshape(-1))
        if org_dataloader:
            org = org_dataloader.dataset[idx][0].T.reshape(-1)

    else:
        input_sample = data.dataset[idx][0].reshape(-1)
        out_sample = np.array(test_out_dict[data_class][date][idx].reshape(-1))
        if org_dataloader:
            org = org_dataloader.dataset[idx][0].reshape(-1)

        
    raw_index = data.dataset[idx][1][1]
    label = list(data.dataset[idx][1][0]).index(1)

    
    if org_dataloader:
        plt.plot(input_sample, label = "Manipulated Input", color = "red")
        plt.plot(org, label = "Original Input")
        plt.plot(out_sample, label = "Reconstruction")
    else:
        plt.plot(input_sample, label = "Input")
        plt.plot(out_sample, label = "Reconstruction")
        
    plt.legend()
    plt.text(0.77,0.3, "loss : {:.2g}".format(np.square(input_sample-out_sample).mean()), fontsize=25,transform=ax.transAxes, 
             color = 'red')
    
    plt.text(0.04,0.02, "X_RMS", fontsize=13,transform=ax.transAxes)
    plt.text(0.12,0.02, "Y_RMS", fontsize=13,transform=ax.transAxes)
    plt.text(0.2,0.02, "Z_RMS", fontsize=13,transform=ax.transAxes)
    plt.text(0.27,0.02, "Front_SP", fontsize=13,transform=ax.transAxes)
    plt.text(0.345,0.02, "Front_TQ", fontsize=13,transform=ax.transAxes)
    plt.text(0.42,0.02, "Hoist_SP", fontsize=13,transform=ax.transAxes)
    plt.text(0.5,0.02, "Hoist_TQ", fontsize=13,transform=ax.transAxes)
    plt.text(0.58,0.02, "Rear_SP", fontsize=13,transform=ax.transAxes)
    plt.text(0.65,0.02, "Rear_TQ", fontsize=13,transform=ax.transAxes)
    plt.text(0.73,0.02, "Slide_SP", fontsize=13,transform=ax.transAxes)
    plt.text(0.81,0.02, "Slide_TQ", fontsize=13,transform=ax.transAxes)
    plt.text(0.9,0.02, "Yaw", fontsize=13,transform=ax.transAxes)
    plt.xlabel("Sensors({}) * Time Window({})".format(n_features, time_window), fontsize = 18)
    plt.ylabel("Normalized Sensor Value", fontsize = 18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("{}th Sample Data (Raw Index: {}), {}".format(idx, raw_index, state_labels[label]), fontsize=15)
    #plt.title("CVAE, {}th Sample".format(idx), fontsize = 30, pad = 15)
    plt.show()


def plot_sample_all_segment(test_normal, test_abnormal,test_out_dict, test_loss_dict, date, idx, data_class = "ABN", model_type = "FCAE",org_dataloader = None):
    name = date.split("/")[-1]
    print(name)
    fig, ax = plt.subplots(figsize=(14,6))
    if data_class == "N":
        data = test_normal[date]
    else:
        data = test_abnormal[date]
              
    if "RAE" in model_type:
        input_sample = data.dataset[idx][0].T.reshape(-1)
        out_sample = np.array(test_out_dict[data_class][date][idx].T.reshape(-1))
        if org_dataloader:
            org = org_dataloader.dataset[idx][0].T.reshape(-1)

    else:
        input_sample = data.dataset[idx][0].reshape(-1)
        out_sample = np.array(test_out_dict[data_class][date][idx].reshape(-1))
        if org_dataloader:
            org = org_dataloader.dataset[idx][0].reshape(-1)

        
    raw_index = data.dataset[idx][1][1]
    label = list(data.dataset[idx][1][0]).index(1)
    LastNode = data.dataset[idx][1][2]
    NextNode = data.dataset[idx][1][3]
    
    if org_dataloader:
        plt.plot(input_sample, label = "Manipulated Input", color = "red")
        plt.plot(org, label = "Original Input")
        plt.plot(out_sample, label = "Reconstruction")
    else:
        plt.plot(input_sample, label = "Input")
        plt.plot(out_sample, label = "Reconstruction")
        
    plt.legend()
    plt.text(0.77,0.3, "loss : {:.2g}".format(np.square(input_sample-out_sample).mean()), fontsize=25,transform=ax.transAxes, 
             color = 'red')
    
    plt.text(0.04,0.02, "X_RMS", fontsize=13,transform=ax.transAxes)
    plt.text(0.12,0.02, "Y_RMS", fontsize=13,transform=ax.transAxes)
    plt.text(0.2,0.02, "Z_RMS", fontsize=13,transform=ax.transAxes)
    plt.text(0.27,0.02, "Front_SP", fontsize=13,transform=ax.transAxes)
    plt.text(0.345,0.02, "Front_TQ", fontsize=13,transform=ax.transAxes)
    plt.text(0.42,0.02, "Hoist_SP", fontsize=13,transform=ax.transAxes)
    plt.text(0.5,0.02, "Hoist_TQ", fontsize=13,transform=ax.transAxes)
    plt.text(0.58,0.02, "Rear_SP", fontsize=13,transform=ax.transAxes)
    plt.text(0.65,0.02, "Rear_TQ", fontsize=13,transform=ax.transAxes)
    plt.text(0.73,0.02, "Slide_SP", fontsize=13,transform=ax.transAxes)
    plt.text(0.81,0.02, "Slide_TQ", fontsize=13,transform=ax.transAxes)
    plt.text(0.9,0.02, "Yaw", fontsize=13,transform=ax.transAxes)
    plt.xlabel("Sensors({}) * Time Window({})".format(n_features, time_window), fontsize = 18)
    plt.ylabel("Normalized Sensor Value", fontsize = 18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.title("{}th Sample Data (Raw Index: {}), {}, ({},{})".format(idx, raw_index, state_labels[label], LastNode, NextNode), fontsize=15)
    #plt.title("CVAE, {}th Sample".format(idx), fontsize = 30, pad = 15)
    plt.show()
    
def plot_pca(z, label, model_type):
    plt.figure(figsize=(10,7))
    colors = ["black","indianred","darkorange","darkgreen","darkblue","salmon","orange","green","blue",]
    state_labels = {0:"stop", 1:"str+noFOUP", 2:"crv+noFOUP", 
                              3:"STB+noFOUP", 4:"Stn+noFOUP", 
                              5:"str+FOUP", 6:"crv+FOUP", 
                              7:"STB+FOUP", 8:"Stn+FOUP"}
    z_pca = PCA(n_components=2).fit_transform(z)
    count = np.zeros(9)
    for i in range(len(z)):
        # 4sate로 변환시
#         if label[i] not in [4,8]:
#             l = label[i] %4
#         elif label[i] ==8:
#             l = 4
        l = label[i]
        if count[l] == 0:
            if l <=4:
                plt.plot(z_pca[i,0], z_pca[i,1], "x", color = colors[l], label = state_labels[l])
            else:
                plt.plot(z_pca[i,0], z_pca[i,1], "o", color = colors[l], label = state_labels[l])
            count[l] += 1
        else:
            if l <=4:
                plt.plot(z_pca[i,0], z_pca[i,1], "x", color = colors[l])
            else:
                plt.plot(z_pca[i,0], z_pca[i,1], "o", color = colors[l])
            count[l] += 1
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize = 15)
    plt.xlim(z_pca[:, 0].min()-0.5, z_pca[:, 0].max()+0.5) # 최소, 최대
    plt.ylim(z_pca[:, 1].min()-0.5, z_pca[:, 1].max()+0.5) # 최소, 최대
    plt.title("PCA Embedding(n=2) of Latent Vectors from {}".format(model_type), fontsize = 20, pad =15)
    plt.show()
    return z_pca


def plot_tsne(z, label, model_type):
    plt.figure(figsize=(10,7))
    colors = ["black","indianred","darkorange","darkgreen","darkblue","salmon","orange","green","blue"]
    state_labels = {0:"stop", 1:"str+noFOUP", 2:"crv+noFOUP", 
                              3:"STB+noFOUP", 4:"Stn+noFOUP", 
                              5:"str+FOUP", 6:"crv+FOUP", 
                              7:"STB+FOUP", 8:"Stn+FOUP"}
    z_tsne = TSNE(n_components=2).fit_transform(z)
    count = np.zeros(9)
    for i in range(len(z)):
        # 4sate로 변환시
#         if label[i] not in [4,8]:
#             l = label[i] %4
#         elif label[i] ==8:
#             l = 4
        l = label[i]
        if count[l] == 0:
            if l <=4:
                plt.plot(z_tsne[i,0], z_tsne[i,1], "x", color = colors[l], label = state_labels[l])
            else:
                plt.plot(z_tsne[i,0], z_tsne[i,1], "o", color = colors[l], label = state_labels[l])
            count[l] += 1
        else:
            if l <=4:
                plt.plot(z_tsne[i,0], z_tsne[i,1], "x", color = colors[l])
            else:
                plt.plot(z_tsne[i,0], z_tsne[i,1], "o", color = colors[l])
            count[l] += 1
    plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5), fontsize = 15)
    plt.xlim(z_tsne[:, 0].min()-5, z_tsne[:, 0].max()+5) # 최소, 최대
    plt.ylim(z_tsne[:, 1].min()-5, z_tsne[:, 1].max()+5) # 최소, 최대
    plt.title("T-SNE Embedding(n=2) of Latent Vectors from {}".format(model_type), fontsize = 20, pad =15)
    plt.show()
    return z_tsne
