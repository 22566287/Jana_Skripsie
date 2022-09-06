# This function prepares the wav files and extracts features that can be used for machine
# learning purposes. The first four features are extracted and the information are stored
# in csv files, ready for import to the classifier.

from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
from functions import *

framelength = 32768 #(2^14 = 16384) or 2^15 = 32768
frameskip = 2000
minfreq = 0
maxfreq = 14000
f0 = 0.0
energy = 0
numberofcompr = 15


#read data names from folder
# Get the list of all files and directories
pathOfFiles = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinPiece"
files = os.listdir(pathOfFiles)


fs = 96000
#Read data from directory with specific file
path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinPiece"
for i in range(1):
    input_data = read(path + "\\" + files[i])
    audio = input_data[1]
    #print("Original audio size: " + str(audio.shape[0]))

    print(i)
    standard = 440
    note = 'A'  
    print(note)

    #Reduce the frequency resolution and take the FFT
    AavPDS = get_average_pds(audio,framelength,frameskip)
    AfreqX = np.fft.fftfreq(len(AavPDS), 1/fs)

    #Remove negative frequency parts and remove unnecessary data
    AfreqXshort = np.zeros(7000)
    AavPDSshort = np.zeros(7000)
    for j in range(7000):
        AavPDSshort[j] = AavPDS[j]
        AfreqXshort[j] = AfreqX[j]

    #normalise to compensate for loudness        
    AavPDSshort = AavPDSshort/max(AavPDSshort)  

    #Find the pitch of the current audio file
    f0A = pitch_detection(AavPDS, AfreqX, numberofcompr,440)
    f0D = pitch_detection(AavPDS, AfreqX, numberofcompr,293.66)
    f0E = pitch_detection(AavPDS, AfreqX, numberofcompr,659.26)
    f0G = pitch_detection(AavPDS, AfreqX, numberofcompr,196)

    

    #Calculate features
    f0val = [f0A,f0D,f0E,f0G]
    noteval = ['A', 'D', 'E', 'G']
    totalE1 = 0
    totalE2 = 0
    totalE3 = 0
    totalE4 = 0
    for k in range(4):
        energy1 = getEnergyInHarmonic(AavPDSshort, f0val[i], 1,framelength, noteval[i])
        totalE1 = totalE1 + energy1

        energy2 = getEnergyInHarmonic(AavPDSshort, f0val[i], 2,framelength, noteval[i])
        totalE2 = totalE2 + energy2

        energy3 = getEnergyInHarmonic(AavPDSshort, f0val[i], 3,framelength, noteval[i])
        totalE3 = totalE3 + energy3

        energy4 = getEnergyInHarmonic(AavPDSshort, f0val[i], 4,framelength, noteval[i])
        totalE4 = totalE4 + energy4
   
    print(totalE1)
    print(totalE2)
    print(totalE3)
    print(totalE4)


#saveToExcelFile('A4.xlsx', 'A4.csv', feat1A, feat2A, feat3A, feat4A)
plotFFTs(AfreqXshort,AavPDSshort, 7000)





