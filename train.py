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

BATCH_SIZE = 16


def eval(model,test_test):
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
        #print(np_pred,np_gt)
        if np_gt * np_pred <0:
            false_cnt = false_cnt +1
    return 1-float(false_cnt)/len(test_set)


#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
# sns.get_dataset_names()
# flight_data = sns.load_dataset("flights")
#print(flight_data.head())
#print(flight_data.shape)



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





dbfile = open('train.pkl', 'rb')      
train_test_set = pickle.load(dbfile) 
dbfile.close() 

#split data 0.8:0.2
fract = 0.8
split_idx = int(fract*len(train_test_set))
train_set = train_test_set[0:split_idx]
test_set = train_test_set[split_idx:]


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print(model)


epochs = 15
for i in range(epochs):
    #for seq, labels in train_inout_seq:
    random.shuffle(train_set)
    
    for cnt in range(0,len(train_set)):
        data = torch.FloatTensor(train_set[cnt][0]).view(-1)
        y_gt = torch.FloatTensor(train_set[cnt][1]).view(-1)
        
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(data)
        #todo: assign correct output 
        single_loss = loss_function(y_pred, y_gt)
        single_loss.backward()
        optimizer.step()

        if cnt%200 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
            test_accuracy = eval(model,test_set)
            print(test_accuracy)

PATH = "entire_model.pt"
# Save
torch.save(model, PATH)


