import pyaudio
import wave
import sys
import numpy as np
import os
import pickle
import random
DATA_SIZE = 32

CHUNK = 2048
SKIP_EVERY = 1
RATE = 44100
TARGET  = "test"

np.set_printoptions(precision=8,suppress=True)
#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/


def gen_set_from_file(file_dir,data_set,good = True):
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
        np_data = np_data[0 : SKIP_EVERY * DATA_SIZE : SKIP_EVERY]
        
        label = np_data[0:1]
        if (good):
            label[0] = 1.0
        else:
            label[0] = -1.0
        data_set.append([np_data,label])
        # stop stream (4)
    stream.stop_stream()
    stream.close()
    # close PyAudio (5)
    p.terminate()

true_train_set = []

#collect data for music
music_dir = './music_'+TARGET+"/"
music_list = os.listdir(music_dir)

for m in music_list:
    fn = music_dir + m
    print(fn)
    gen_set_from_file(fn,true_train_set)

#collect data for none-music
false_train_set = []
music_dir = './none_music_'+TARGET+"/"
music_list = os.listdir(music_dir)
for m in music_list:
    fn = music_dir + m
    gen_set_from_file(fn,false_train_set,False)


min_len = min(len(false_train_set),len(true_train_set))
true_train_set = true_train_set[0:min_len]
false_train_set = false_train_set[0:min_len]
data_set = true_train_set + false_train_set

random.shuffle(data_set)

dbfile = open(TARGET+'.pkl', 'wb') 
# source, destination 
pickle.dump(data_set, dbfile)                      
dbfile.close()

dbfile = open(TARGET+'.pkl', 'rb')      
train_test_set = pickle.load(dbfile) 
dbfile.close() 

print(len(train_test_set))
for data in train_test_set:
    print(data[1])
    continue

