from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from functions import *
import math

framelength = math.pow(2,14)
frameskip = 1000
minfreq = 0
maxfreq = 14000
f0 = 0.0

#read data from folder
input_data = read("./data/A1-A4-001.wav")
fs = input_data[0]
audio = input_data[1]

#Get fft of audio
audiofft = abs(np.fft.fft(audio))
freqX = np.fft.fftfreq(len(audiofft), 1/fs)
print(audiofft.shape)
f0 = pitch_detection(audiofft, freqX, 3)
plt.plot(freqX,audiofft) #plot freqX vs Hx

#plt.plot(audio[0:400000])
plt.title("FFT")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.xlim(0, 14000)
plt.show()




#yA = get_average_pds(audio,framelength,frameskip)