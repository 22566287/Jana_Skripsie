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
standard = 196
#audioArr = np.array(["A1-G3-001.wav", "A1-D4-001.wav","A1-A4-001.wav","A1-E5-001.wav","A1-E6-001.wav"])
africa1 = read("./data/A1-A4-001.wav")
africa2 = read("./data/A2-A4-001.wav")
conv1 = read("./data/C1-A4-001.wav")
conv2 = read("./data/C2-A4-001.wav")

input_data = read("./data/A2-A4-001.wav")
fs = input_data[0]
audio = input_data[1]
print("Original audio size: " + str(audio.shape[0]))

#Reduce the frequency resolution and take the FFT
avPDS = get_average_pds(audio,framelength,frameskip)
print("Reduced frequency resolution audio size: " + str(avPDS.shape[0]))
freqX = np.fft.fftfreq(len(avPDS), 1/fs)

#Find the pitch of the current audio file
f0 = pitch_detection(avPDS, freqX, 15)
print("True expected frequency: " + str(standard) + " Hz")
print("Pitch detector result: " + str(f0) + " Hz")

#Plot the FFT with reduced frequency resolution
avPDS = avPDS/max(avPDS)          #normalise to compensate for loudness
plt.plot(freqX,avPDS) 
plt.title("FFT")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.xlim(0, 7000)
plt.grid()
plt.show()

energy = getEnergyInHarmonic(avPDS, f0, 1)
print(energy)



#Africa 1 violin calculations - subplot1
avPDSA1 = get_average_pds(africa1[1],framelength,frameskip)
freqXA1 = np.fft.fftfreq(len(avPDSA1), 1/fs)
f0A1 = pitch_detection(avPDSA1, freqXA1, 15)
avPDSA1 = avPDSA1/max(avPDSA1)          
energyA1 = getEnergyInHarmonic(avPDSA1, f0A1, 2)

#Africa 2 violin calculations - subplot2
avPDSA2 = get_average_pds(africa2[1],framelength,frameskip)
freqXA2 = np.fft.fftfreq(len(avPDSA2), 1/fs)
f0A2 = pitch_detection(avPDSA2, freqXA2, 15)
avPDSA2 = avPDSA2/max(avPDSA2)          
energyA2 = getEnergyInHarmonic(avPDSA2, f0A2, 1)

#Conventional 1 violin calculations - subplot3
avPDSC1 = get_average_pds(conv1[1],framelength,frameskip)
freqXC1 = np.fft.fftfreq(len(avPDSC1), 1/fs)
f0C1 = pitch_detection(avPDSC1, freqXC1, 15)
avPDSC1 = avPDSC1/max(avPDSC1)          
energyC1 = getEnergyInHarmonic(avPDSC1, f0C1, 2)

#Conventional 2 violin calculations - subplot4
avPDSC2 = get_average_pds(conv2[1],framelength,frameskip)
freqXC2 = np.fft.fftfreq(len(avPDSC2), 1/fs)
f0C2 = pitch_detection(avPDSC2, freqXC2, 15)
avPDSC2 = avPDSC2/max(avPDSC2)          
energyC2 = getEnergyInHarmonic(avPDSC2, f0C2, 1)



fig = plt.figure(figsize=(6, 4))
t = np.arange(-5.0, 1.0, 0.1)

sub1 = fig.add_subplot(221) # instead of plt.subplot(2, 2, 1)
sub1.set_title('A1A4' + str(energyA1)) # non OOP: plt.title('The function f')
sub1.set_xlim(0, 7000)
sub1.plot(freqXA1,avPDSA1)

sub2 = fig.add_subplot(222)
sub2.set_title('A2A4' + str(energyA2))
sub2.set_xlim(0, 7000)
sub2.plot(freqXA2,avPDSA2)

sub3 = fig.add_subplot(223)
sub3.set_title('C1A4' + str(energyC1))
sub3.set_xlim(0, 7000)
sub3.plot(freqXC1,avPDSC1)

sub4 = fig.add_subplot(224)
sub4.set_title('C2A4' + str(energyC2))
sub4.set_xlim(0, 7000)
sub4.plot(freqXC2,avPDSC2)

plt.tight_layout()
plt.show()
