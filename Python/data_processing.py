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
africa1 = [""]*5
africa2 = [""]*5
conv1 = [""]*5
conv2 = [""]*5
conv3 = [""]*5
conv4 = [""]*5
conv5 = [""]*5
conv6 = [""]*5
conv7 = [""]*5
conv8 = [""]*5
conv9 = [""]*5
conv10 = [""]*5
fact1 = [""]*5
fact2 = [""]*5
fact3 = [""]*5



# Standard frequency values:
# G3 = 196 Hz
# D4 = 293.66 Hz
# A4 = 440 Hz
# E5 = 659.26 Hz
# E6 = 1318.51 Hz

#read data names from folder
files = readInData()

for i in range(5):
    africa1[i] = files[i]
    africa2[i] = files[5+i]
    conv1[i] = files[10+i]
    conv10[i] = files[15+i]
    conv2[i] = files[20+i]
    conv3[i] = files[25+i]
    conv4[i] = files[30+i]
    conv5[i] = files[35+i]
    conv6[i] = files[40+i]
    conv7[i] = files[45+i]
    conv8[i] = files[50+i]
    conv9[i] = files[55+i]
    fact1[i] = files[60+i]
    fact2[i] = files[65+i]
    fact3[i] = files[70+i]

print(africa1[0])
print(africa2[0])
print(conv1[0])
print(conv2[0])
print(conv3[0])
print(conv4[0])
print(conv5[0])
print(conv10[0])
print(fact1[0])
print(fact3[4])

#try to read data from directory with specific file
path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData"
readintry1 = read(path + "\\" + africa1[0])
print(readintry1[0])
print(readintry1[1])

standard = 440
note = 'G'
input_data = read("./data/A1-G3-001.wav")
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

#Plot the FFT with reduced frequency resolution
avPDS = avPDS/max(avPDS)          #normalise to compensate for loudness
avPDSshort = avPDSshort/max(avPDSshort)   
plt.plot(freqXshort,avPDSshort) 
plt.title("FFT")
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.grid()
plt.show()

energy1 = getEnergyInHarmonic(avPDSshort, f0, 1,framelength, note)
energy2 = getEnergyInHarmonic(avPDSshort, f0, 2,framelength, note)
energy3 = getEnergyInHarmonic(avPDSshort, f0, 3,framelength, note)
energy4 = getEnergyInHarmonic(avPDSshort, f0, 4,framelength, note)





