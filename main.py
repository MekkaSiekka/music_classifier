import pyaudio
import wave
import sys
import numpy as np
import os
CHUNK = 1024
#https://stackabuse.com/time-series-prediction-using-lstm-with-pytorch-in-python/


# if len(sys.argv) < 2:
#     print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
#     sys.exit(-1)
music_src_dir = '/Users/caojiang/Music/QQ音乐/'
lists = os.listdir(music_src_dir)
print(lists)
wf = wave.open('/Users/caojiang/Downloads/file_example_WAV_2MG.wav', 'rb')

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
while len(data) > 0:
   # print(data)
    np_data =  np.fromstring(data,dtype=np.int16)
    print(np_data)
    print(np_data.shape)
    #write the data for playing
    stream.write(data)
    data = wf.readframes(CHUNK)
    cnt = cnt+1
    print(cnt)

# stop stream (4)
stream.stop_stream()
stream.close()

# close PyAudio (5)
p.terminate()