from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from functions import *
import math

framelength = math.pow(2,14)
frameskip = 1000
minfreq = 0
maxfreq = 14000

#plt.rcParams["figure.figsize"] = [7.50, 3.50]
#plt.rcParams["figure.autolayout"] = True
input_data = read("./A1-A4-001.wav")
fs = input_data[0]
audio = input_data[1]
for i in range(10):
    print(audio[i])
print(fs)
plt.plot(audio[0:400000])
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.show()

yA = get_average_pds(audio,framelength,frameskip)