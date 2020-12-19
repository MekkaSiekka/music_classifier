import matplotlib.pyplot as plt

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

def gen_color():
    r = random.random()
    b = random.random()
    g = random.random()
    color = (r, g, b)
    return color
def plot(file_name,train_label,test_label):
    dbfile = open(file_name, 'rb')      
    log_data = pickle.load(dbfile) 
    dbfile.close() 

    x = log_data["epoch_acc"]
    y1 = log_data["train_acc"]
    y2 = log_data["test_acc"]

    loss1 = log_data["train_loss"]
    loss2 = log_data["test_loss"]

    plt.plot(x, y1,'-k', color= gen_color(),label=train_label)
    plt.plot(x, y2,'-k', color= gen_color(),label=test_label)
    plt.xlabel("epoches")
    plt.ylabel("accuracy")
    plt.title("accuracy during training for different data generation methods")



plot("log.pkl","train accuracy even 32","test accuray even 32")
plot("log_none_skip.pkl","train accuracy first 32","test accuray first 32")

plt.legend()
plt.show()