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


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]



PATH = "entire_model.pt"


dbfile = open('train_test_pickle.pkl', 'rb')      
train_test_set = pickle.load(dbfile) 
dbfile.close() 

fract = 0.8
split_idx = int(fract*len(train_test_set))
train_set = train_test_set[0:split_idx]
test_set = train_test_set[split_idx:]



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
    if np_gt * np_pred <0:
        false_cnt = false_cnt +1

print("ACCURATE RATE", 1-float(false_cnt)/len(test_set))
