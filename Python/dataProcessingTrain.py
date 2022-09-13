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

feat1A, feat2A, feat3A, feat4A = [0]*72, [0]*72, [0]*72, [0]*72
feat1D, feat2D, feat3D, feat4D = [0]*72, [0]*72, [0]*72, [0]*72
feat1E, feat2E, feat3E, feat4E = [0]*72, [0]*72, [0]*72, [0]*72
feat1G, feat2G, feat3G, feat4G = [0]*72, [0]*72, [0]*72, [0]*72



#read data names from folder
files = readInData()
print(files)

violinCounter = -1
noteCounter = -1
fs = 96000
audioArr = [0,0,0,0]
#Read data from directory with specific file
path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\trainingSet"
for i in range(72):
    input_data = read(path + "\\" + files[i])
    audio = input_data[1]
    #print("Original audio size: " + str(audio.shape[0]))

    #Split training set into 4 vectors instead of one for better results in classifier
    audioArr = timeFrame(audio, audioArr,4)

  
        
    if(i%4 == 0):
        violinCounter = violinCounter + 1
        noteCounter = -1
        for k in range(4):
            noteCounter = noteCounter + 1
            print(i)
            standard = 440
            note = 'A'  
            print(note)

            #Reduce the frequency resolution and take the FFT
            AavPDS = get_average_pds(audioArr[k],framelength,frameskip)
            AfreqX = np.fft.fftfreq(len(AavPDS), 1/fs)

            #Remove negative frequency parts and remove unnecessary data
            AfreqXshort = np.zeros(7000)
            AavPDSshort = np.zeros(7000)
            for j in range(7000):
                AavPDSshort[j] = AavPDS[j]
                AfreqXshort[j] = AfreqX[j]

            #Find the pitch of the current audio file
            f0 = pitch_detection(AavPDS, AfreqX, numberofcompr,standard)

            #normalise to compensate for loudness        
            AavPDSshort = AavPDSshort/max(AavPDSshort)   

            #Calculate features
            feat1A[4*violinCounter + noteCounter] = getEnergyInHarmonic(AavPDSshort, f0, 1,framelength, note)
            feat2A[4*violinCounter + noteCounter] = getEnergyInHarmonic(AavPDSshort, f0, 2,framelength, note)
            feat3A[4*violinCounter + noteCounter] = getEnergyInHarmonic(AavPDSshort, f0, 3,framelength, note)
            feat4A[4*violinCounter + noteCounter] = getEnergyInHarmonic(AavPDSshort, f0, 4,framelength, note)

            

    if(i%4 == 1):
        noteCounter = -1
        for k in range(4):
            noteCounter = noteCounter + 1
            standard = 293.66
            note = 'D' 
            print(note) 

            #Reduce the frequency resolution and take the FFT
            DavPDS = get_average_pds(audioArr[k],framelength,frameskip)
            DfreqX = np.fft.fftfreq(len(DavPDS), 1/fs)

            #Remove negative frequency parts and remove unnecessary data
            DfreqXshort = np.zeros(7000)
            DavPDSshort = np.zeros(7000)
            for j in range(7000):
                DavPDSshort[j] = DavPDS[j]
                DfreqXshort[j] = DfreqX[j]

            #Find the pitch of the current audio file
            f0 = pitch_detection(DavPDS, DfreqX, numberofcompr, standard)

            #normalise to compensate for loudness        
            DavPDSshort = DavPDSshort/max(DavPDSshort)   

            #Calculate features
            feat1D[4*violinCounter + noteCounter] = getEnergyInHarmonic(DavPDSshort, f0, 1,framelength, note)
            feat2D[4*violinCounter + noteCounter] = getEnergyInHarmonic(DavPDSshort, f0, 2,framelength, note)
            feat3D[4*violinCounter + noteCounter] = getEnergyInHarmonic(DavPDSshort, f0, 3,framelength, note)
            feat4D[4*violinCounter + noteCounter] = getEnergyInHarmonic(DavPDSshort, f0, 4,framelength, note)

    if(i%4 == 2):
        noteCounter = -1
        for k in range(4):
            noteCounter = noteCounter + 1
            standard = 659.26
            note = 'E'  
            print(note)

            #Reduce the frequency resolution and take the FFT
            EavPDS = get_average_pds(audioArr[k],framelength,frameskip)
            EfreqX = np.fft.fftfreq(len(EavPDS), 1/fs)

            #Remove negative frequency parts and remove unnecessary data
            EfreqXshort = np.zeros(7000)
            EavPDSshort = np.zeros(7000)
            for j in range(7000):
                EavPDSshort[j] = EavPDS[j]
                EfreqXshort[j] = EfreqX[j]

            #Find the pitch of the current audio file
            f0 = pitch_detection(EavPDS, EfreqX, numberofcompr, standard)

            #normalise to compensate for loudness        
            EavPDSshort = EavPDSshort/max(EavPDSshort)   

            #Calculate features
            feat1E[4*violinCounter + noteCounter] = getEnergyInHarmonic(EavPDSshort, f0, 1,framelength, note)
            feat2E[4*violinCounter + noteCounter] = getEnergyInHarmonic(EavPDSshort, f0, 2,framelength, note)
            feat3E[4*violinCounter + noteCounter] = getEnergyInHarmonic(EavPDSshort, f0, 3,framelength, note)
            feat4E[4*violinCounter + noteCounter] = getEnergyInHarmonic(EavPDSshort, f0, 4,framelength, note)

    # if(i%4 == 3):
    #     standard = 1318.51
    #     note = 'E'  

        # #Reduce the frequency resolution and take the FFT
        # avPDS = get_average_pds(audio,framelength,frameskip)
        # freqX = np.fft.fftfreq(len(avPDS), 1/fs)

        # #Remove negative frequency parts and remove unnecessary data
        # freqXshort = np.zeros(7000)
        # avPDSshort = np.zeros(7000)
        # for j in range(7000):
        #     avPDSshort[j] = avPDS[j]
        #     freqXshort[j] = freqX[j]

        # #Find the pitch of the current audio file
        # f0 = pitch_detection(avPDS, freqX, numberofcompr, standard)

        # #normalise to compensate for loudness        
        # avPDSshort = avPDSshort/max(avPDSshort)   

    if(i%4 == 3):
        noteCounter = -1
        for k in range(4):
            noteCounter = noteCounter + 1
            standard = 196
            note = 'G'
            print(note)

            #Reduce the frequency resolution and take the FFT
            GavPDS = get_average_pds(audioArr[k],framelength,frameskip)
            GfreqX = np.fft.fftfreq(len(GavPDS), 1/fs)

            #Remove negative frequency parts and remove unnecessary data
            GfreqXshort = np.zeros(7000)
            GavPDSshort = np.zeros(7000)
            for j in range(7000):
                GavPDSshort[j] = GavPDS[j]
                GfreqXshort[j] = GfreqX[j]

            #Find the pitch of the current audio file
            f0 = pitch_detection(GavPDS, GfreqX, numberofcompr, standard)

            #normalise to compensate for loudness        
            GavPDSshort = GavPDSshort/max(GavPDSshort)   

            #Calculate features
            feat1G[4*violinCounter + noteCounter] = getEnergyInHarmonic(GavPDSshort, f0, 1,framelength, note)
            feat2G[4*violinCounter + noteCounter] = getEnergyInHarmonic(GavPDSshort, f0, 2,framelength, note)
            feat3G[4*violinCounter + noteCounter] = getEnergyInHarmonic(GavPDSshort, f0, 3,framelength, note)
            feat4G[4*violinCounter + noteCounter] = getEnergyInHarmonic(GavPDSshort, f0, 4,framelength, note)
        
            

    #Find the pitch of the current audio file
    # print("True expected frequency: " + str(standard) + " Hz")
    # print("Pitch detector result: " + str(f0) + " Hz\n")

    
        

saveToExcelFile('A4train.xlsx', 'A4train.csv', feat1A, feat2A, feat3A, feat4A)
saveToExcelFile('D4train.xlsx', 'D4train.csv', feat1D, feat2D, feat3D, feat4D)
saveToExcelFile('E5train.xlsx', 'E5train.csv', feat1E, feat2E, feat3E, feat4E)
saveToExcelFile('G3train.xlsx', 'G3train.csv', feat1G, feat2G, feat3G, feat4G)

    




















