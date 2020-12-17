
import pyaudio
import wave
import sys
import os
import pickle

music_dir = '/Users/caojiang/Music/QQ音乐/'

music_list = os.listdir(music_dir)

print(music_list)

for music_name in music_list:
    abs_dir = music_dir + music_name
    a = "ffmpeg -i "
    b = "-acodec pcm_u8 -ar 22050 "
    new_dir = abs_dir.replace('.mp3','.wav')
    cmd = a + abs_dir + b + new_dir

