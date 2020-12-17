import pyaudio
import wave
import sys
import numpy as np
import os
import pickle
CHUNK = 2048
RATE = 44100

np.set_printoptions(precision=8,suppress=True)
#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/


def gen_set_from_file(file_dir,train_test_set,good = True):
    wf = wave.open(file_dir, 'rb')
    # instantiate PyAudio (1)
    p = pyaudio.PyAudio()
    # open stream (2)
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # read data
    data = wf.readframes(CHUNK)
    prev_size = len(data)
    # play stream (3)
    cnt = 0 
    while len(data) > 0 and len(data)==prev_size:
    # print(data)
        data = wf.readframes(CHUNK)
        np_data =  np.fromstring(data,dtype=np.int16)
        #write the data for playing
        #stream.write(data)
        np_data = np_data.astype(np.float32)
        np_data = np_data/32780
        label = np_data[0:1]
        if (good):
            label[0] = 1.0
        else:
            label[0] = 0.0
        train_test_set.append([np_data,label])
        # stop stream (4)
    stream.stop_stream()
    stream.close()
    # close PyAudio (5)
    p.terminate()

train_test_set = []
gen_set_from_file('./music/CJ-Whoopty.wav',train_test_set)
print(train_test_set)

dbfile = open('train_test_pickle.pkl', 'wb') 
# source, destination 
pickle.dump(train_test_set, dbfile)                      
dbfile.close()

dbfile = open('train_test_pickle.pkl', 'rb')      
train_test_set = pickle.load(dbfile) 
dbfile.close() 

for data in train_test_set:
    print(data[0][0:20])
    continue