from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from functions import *
import math

framelength = 32768 #(2^14 = 16384) or 2^15 = 32768
frameskip = 4000
minfreq = 0
maxfreq = 14000
f0 = 0.0

# Standard frequency values:
# G3 = 196 Hz
# D4 = 293.66 Hz
# A4 = 440 Hz
# E5 = 659.26 Hz
# E6 = 1318.51 Hz

#read data from folder
standard = 196
audioArr = np.array(["A1-G3-001.wav", "A1-D4-001.wav","A1-A4-001.wav","A1-E5-001.wav","A1-E6-001.wav"])
input_data = read("./data/A1-G3-001.wav")
fs = input_data[0]
audio = input_data[1]
print("Original audio size: " + str(audio.shape[0]))

#Get fft of audio with reduced frequency
yA = get_average_pds(audio,framelength,frameskip)
print("Reduced frequency resolution audio size: " + str(yA.shape[0]))

# audiofft = abs(np.fft.fft(audio))
# freqX = np.fft.fftfreq(len(audiofft), 1/fs)
# f0 = pitch_detection(audiofft, freqX, 10)
freqX = np.fft.fftfreq(len(yA), 1/fs)
f0 = pitch_detection(yA, freqX, 15)
print("True expected frequency: " + str(standard) + " Hz")
print("Pitch detector result: " + str(f0) + " Hz")
plt.plot(freqX,yA) #plot freqX vs Hx

#plt.plot(audio[0:400000])
plt.title("FFT")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.xlim(0, 14000)
plt.show()


