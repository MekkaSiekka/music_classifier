import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CHUNK = 1024

import pyaudio
import wave
import sys
import os

import pickle #load train data

import random #shuffle



dbfile = open('rnn.pkl', 'rb')      
model = pickle.load(dbfile) 
dbfile.close() 

dbfile = open('train_test_pickle.pkl', 'rb')      
train_test_set = pickle.load(dbfile) 
dbfile.close() 


for cnt in range(0,5):
        data = torch.FloatTensor(train_test_set[0][0]).view(-1)
        y_gt = torch.FloatTensor(train_test_set[0][1]).view(-1)
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        y_pred = model(data)
        print(y_pred)


