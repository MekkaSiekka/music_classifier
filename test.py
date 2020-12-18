import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pyaudio
import wave

import sys
import os

import pickle #load train data

import random #shuffle

from model import LSTM

def eval(model,test_test):
    for cnt in range(len(test_set)):
        data = torch.FloatTensor(test_set[cnt][0]).view(-1)
        y_gt = torch.FloatTensor(test_set[cnt][1]).view(-1) 
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(data)
        #running test
        np_pred = y_pred[0].detach().numpy()
        np_gt = y_gt[0].detach().numpy()
        #print(np_pred,np_gt)
        if np_gt * np_pred <0:
            false_cnt = false_cnt +1
    return 1-float(false_cnt)/len(test_set)

dbfile = open('test.pkl', 'rb')      
train_test_set = pickle.load(dbfile) 
dbfile.close() 

fract = 0.8
split_idx = int(fract*len(train_test_set))
train_set = train_test_set[0:split_idx]
test_set = train_test_set[split_idx:]

PATH = "entire_model.pt"
model = torch.load(PATH)

false_cnt = 0
for cnt in range(len(test_set)):
    data = torch.FloatTensor(test_set[cnt][0]).view(-1)
    y_gt = torch.FloatTensor(test_set[cnt][1]).view(-1) 
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
    y_pred = model(data)
    #running test
    np_pred = y_pred[0].detach().numpy()
    np_gt = y_gt[0].detach().numpy()
    print(np_pred,np_gt)
    if np_gt * np_pred <0:
        false_cnt = false_cnt +1

print("ACCURATE RATE", 1-float(false_cnt)/len(test_set))
