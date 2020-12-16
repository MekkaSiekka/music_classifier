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


#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/
# sns.get_dataset_names()
# flight_data = sns.load_dataset("flights")
#print(flight_data.head())
#print(flight_data.shape)


music_src_dir = '/Users/caojiang/Music/QQ音乐/'
lists = os.listdir(music_src_dir)
print(lists)
wf = wave.open('./file_example_WAV_2MG.wav', 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

wave_data = wf.readframes(CHUNK)
np_wave_data =  np.fromstring(wave_data,dtype=np.int16)
#INPUT_SIZE = np_wave_data.shape[0]


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
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)


epochs = 150

data = torch.FloatTensor(np_wave_data).view(-1)

for i in range(epochs):
    #for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(data[0:10])

        single_loss = loss_function(y_pred, data[1:2])
        single_loss.backward()
        optimizer.step()

        if i%25 == 1:
            print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')