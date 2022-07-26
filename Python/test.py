from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from functions import *

framelength = 32768 #(2^14 = 16384) or 2^15 = 32768
frameskip = 4000
minfreq = 0
maxfreq = 14000
f0 = 0.0
energy = 0

# Standard frequency values:
# G3 = 196 Hz
# D4 = 293.66 Hz
# A4 = 440 Hz
# E5 = 659.26 Hz
# E6 = 1318.51 Hz

#read data from folder
files = readInData()
print(files[20])
standard = 196
#audioArr = np.array(["A1-G3-001.wav", "A1-D4-001.wav","A1-A4-001.wav","A1-E5-001.wav","A1-E6-001.wav"])

input_data = read("./data/A2-A4-001.wav")
fs = input_data[0]
audio = input_data[1]
print("Original audio size: " + str(audio.shape[0]))

#Reduce the frequency resolution and take the FFT
avPDS = get_average_pds(audio,framelength,frameskip)
print("Reduced frequency resolution audio size: " + str(avPDS.shape[0]))
freqX = np.fft.fftfreq(len(avPDS), 1/fs)

#Remove negative frequency parts and remove unnecessary data
freqXshort = np.zeros(5500)
avPDSshort = np.zeros(5500)
for i in range(5500):
    avPDSshort[i] = avPDS[i]
    freqXshort[i] = freqX[i]



#Find the pitch of the current audio file
f0 = pitch_detection(avPDS, freqX, 15)
print("True expected frequency: " + str(standard) + " Hz")
print("Pitch detector result: " + str(f0) + " Hz")


#Plot the FFT with reduced frequency resolution and normalised
#avPDS = avPDS/max(avPDS)          #normalise to compensate for loudness
avPDSshort = avPDSshort/max(avPDSshort)   
#plotFFTs(freqX,avPDS,12000)
plotFFTs(freqXshort,avPDSshort,12000)

#Calculate energy in feature bands
energy = getEnergyInHarmonic(avPDSshort, f0, 1, framelength)

#save data to a text file
#saveToTextFile("A1E6.txt", avPDS)

print("Done")


