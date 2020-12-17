import pyaudio
import wave
import sys
import numpy as np
import os
import pickle
CHUNK = 2048
#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/


# if len(sys.argv) < 2:
#     print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
#     sys.exit(-1)
music_src_dir = '/Users/caojiang/Music/QQ音乐/'
lists = os.listdir(music_src_dir)
print(lists)
wf = wave.open('./music/'+lists[0], 'rb')

# instantiate PyAudio (1)
p = pyaudio.PyAudio()

# open stream (2)
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True)

# read data
data = wf.readframes(CHUNK)
# play stream (3)
cnt = 0 
train_test_set = []

while len(data) > 0 and len(data)==CHUNK * 2 :
    # print(data)
    print(len(data))
    np_data =  np.fromstring(data,dtype=np.int16)
    print(np_data)
    print(np_data.shape)
    #write the data for playing
    #stream.write(data)
    data = wf.readframes(CHUNK)
    cnt = cnt+1
    print(cnt)
    np_wave_data =  np.fromstring(data,dtype=np.int16)
    label = np_wave_data[0:1]
    label[0] = 1
    train_test_set.append([np_data,label])
# stop stream (4)
stream.stop_stream()
stream.close()
# close PyAudio (5)
p.terminate()

dbfile = open('train_test_pickle.pkl', 'ab') 
# source, destination 
pickle.dump(train_test_set, dbfile)                      
dbfile.close()

dbfile = open('train_test_pickle.pkl', 'rb')      
train_test_set = pickle.load(dbfile) 
dbfile.close() 

for data in train_test_set:
    print(data[1])