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

DATA_SIZE = 32
SKIP_EVERY = 4
CHUNK = 2048
RATE = 44100


p=pyaudio.PyAudio() # start the PyAudio class
stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
              frames_per_buffer=CHUNK) #uses default input device

PATH = "entire_model.pt"
model = torch.load(PATH)


# create a numpy array holding a single read of audio data
for i in range(1000): #to it a few times just to see
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    np_data =  np.fromstring(data,dtype=np.int16)
    np_data = np_data.astype(np.float32)
    np_data = np_data/32780
    c = np.empty((np_data.size * 2), dtype=np_data.dtype)
    c[0::2] = np_data
    c[1::2] = np_data
    np_data = c
    np_data = np_data[0 : SKIP_EVERY * DATA_SIZE : SKIP_EVERY]
    #print(np_data)

    data = torch.FloatTensor(np_data).view(-1)
    model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
    y_pred = model(data)
    print(y_pred)

# close the stream gracefully
stream.stop_stream()
stream.close()
p.terminate()