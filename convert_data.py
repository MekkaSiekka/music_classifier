# converts the data from .mp3 to wav, given a director
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
    abs_dir = '\''+abs_dir+'\''
    a = "ffmpeg -i "
    b = " -acodec  pcm_s16le -ar 16000 "
    new_dir = abs_dir.replace('.mp3','.wav')
    cmd = a + abs_dir + b + new_dir
    print(cmd)
    os.system(cmd)

os.system('cp -r ~/Music/QQ音乐/* ./music')




