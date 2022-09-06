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

feat1A = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat2A = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat3A = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat4A = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

feat1D = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat2D = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat3D = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat4D = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

feat1E = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat2E = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat3E = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat4E = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

feat1G = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat2G = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat3G = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat4G = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


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

violinCounter = -1
fs = 96000
#Read data from directory with specific file
path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData"
for i in range(75):
    input_data = read(path + "\\" + files[i])
    audio = input_data[1]
    #print("Original audio size: " + str(audio.shape[0]))

  
        
    if(i%5 == 0):
        violinCounter = violinCounter + 1
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

        #Find the pitch of the current audio file
        f0 = pitch_detection(AavPDS, AfreqX, numberofcompr,standard)

        #normalise to compensate for loudness        
        AavPDSshort = AavPDSshort/max(AavPDSshort)   

        #Calculate features
        feat1A[violinCounter] = getEnergyInHarmonic(AavPDSshort, f0, 1,framelength, note)
        feat2A[violinCounter] = getEnergyInHarmonic(AavPDSshort, f0, 2,framelength, note)
        feat3A[violinCounter] = getEnergyInHarmonic(AavPDSshort, f0, 3,framelength, note)
        feat4A[violinCounter] = getEnergyInHarmonic(AavPDSshort, f0, 4,framelength, note)

        

    if(i%5 == 1):
        standard = 293.66
        note = 'D' 
        print(note) 

        #Reduce the frequency resolution and take the FFT
        DavPDS = get_average_pds(audio,framelength,frameskip)
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
        feat1D[violinCounter] = getEnergyInHarmonic(DavPDSshort, f0, 1,framelength, note)
        feat2D[violinCounter] = getEnergyInHarmonic(DavPDSshort, f0, 2,framelength, note)
        feat3D[violinCounter] = getEnergyInHarmonic(DavPDSshort, f0, 3,framelength, note)
        feat4D[violinCounter] = getEnergyInHarmonic(DavPDSshort, f0, 4,framelength, note)

    if(i%5 == 2):
        standard = 659.26
        note = 'E'  
        print(note)

        #Reduce the frequency resolution and take the FFT
        EavPDS = get_average_pds(audio,framelength,frameskip)
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
        feat1E[violinCounter] = getEnergyInHarmonic(EavPDSshort, f0, 1,framelength, note)
        feat2E[violinCounter] = getEnergyInHarmonic(EavPDSshort, f0, 2,framelength, note)
        feat3E[violinCounter] = getEnergyInHarmonic(EavPDSshort, f0, 3,framelength, note)
        feat4E[violinCounter] = getEnergyInHarmonic(EavPDSshort, f0, 4,framelength, note)

    if(i%5 == 3):
        standard = 1318.51
        note = 'E'  

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

    if(i%5 == 4):
        standard = 196
        note = 'G'
        print(note)

        #Reduce the frequency resolution and take the FFT
        GavPDS = get_average_pds(audio,framelength,frameskip)
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
        feat1G[violinCounter] = getEnergyInHarmonic(GavPDSshort, f0, 1,framelength, note)
        feat2G[violinCounter] = getEnergyInHarmonic(GavPDSshort, f0, 2,framelength, note)
        feat3G[violinCounter] = getEnergyInHarmonic(GavPDSshort, f0, 3,framelength, note)
        feat4G[violinCounter] = getEnergyInHarmonic(GavPDSshort, f0, 4,framelength, note)
    
        

    #Find the pitch of the current audio file
    # print("True expected frequency: " + str(standard) + " Hz")
    # print("Pitch detector result: " + str(f0) + " Hz\n")

    
        


#saveToTextFile("f3freq.txt", freqArr)
#plot4FFTs(freqXshort, AavPDSshort, DavPDSshort, EavPDSshort, GavPDSshort)


saveToExcelFile('A4.xlsx', 'A4.csv', feat1A, feat2A, feat3A, feat4A)
saveToExcelFile('D4.xlsx', 'D4.csv', feat1D, feat2D, feat3D, feat4D)
saveToExcelFile('E5.xlsx', 'E5.csv', feat1E, feat2E, feat3E, feat4E)
saveToExcelFile('G3.xlsx', 'G3.csv', feat1G, feat2G, feat3G, feat4G)

    













#Plot the FFT with reduced frequency resolution
# plt.plot(freqXshort,avPDSshort) 
# plt.title("FFT")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.xlim(0,7000)
# plt.grid()
# plt.show()






