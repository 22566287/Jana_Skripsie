from ctypes import sizeof
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.io import wavfile
from functions import *
#Comment Code Block Ctrl+K+C/Ctrl+K+U


framelength = math.pow(2,14)
frameskip = 1000
minfreq = 0
maxfreq = 14000

sample_rate, samples = wavfile.read('A1-A4-001.wav')
fs, data = wavfile.read('./A1-A4-001.wav')      #data has dimension 5059256, sample rate is 96kHz
#data = np.array(data)
# print({data.shape[0]})
# length = data.shape[0] / fs
# print(length)

# time = np.linspace(0, fs, data.shape[0])
# plt.plot(time, data[0], label="Left channel")
# plt.xlabel("Time [s]")
# plt.ylabel("Amplitude")
# plt.show()




#get average pds and use FFT
yA = get_average_pds(data,framelength,frameskip)

#Normalise peaks to compensate for different loudness
#yA = yA/max(yA);      





