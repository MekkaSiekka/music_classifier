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



model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
print(model)


dbfile = open('train_test_pickle.pkl', 'rb')      
train_test_set = pickle.load(dbfile) 
dbfile.close() 

fract = 0.8
split_idx = int(fract*len(train_test_set))
train_set = train_test_set[0:split_idx]
test_set = train_test_set[split_idx:]


epochs = 10
for i in range(epochs):
    #for seq, labels in train_inout_seq:
    random.shuffle(train_set)
    #for cnt in range(0,len(train_set)):
    for cnt in range(1):
        data = torch.FloatTensor(train_set[cnt][0]).view(-1)
        y_gt = torch.FloatTensor(train_set[cnt][1]).view(-1)
        print(y_gt)
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(data)
        #todo: assign correct output 
        single_loss = loss_function(y_pred, y_gt)
        single_loss.backward()
        optimizer.step()

        #if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')
        print(cnt)

PATH = "entire_model.pt"
# Save
torch.save(model, PATH)

model2 = torch.load(PATH)

for cnt in range(len(test_set)):
    data = torch.FloatTensor(test_set[cnt][0]).view(-1)
    y_gt = torch.FloatTensor(test_set[cnt][1]).view(-1)
    y_pred = model2(data)
    print(y_pred)
    print(y_gt)
    
