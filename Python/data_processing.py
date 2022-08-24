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
freqArr = np.zeros(5)

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

fs = 96000
#Read data from directory with specific file
path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData"
for i in range(5):
    input_data = read(path + "\\" + conv5[i])
    audio = input_data[1]
    print("Original audio size: " + str(audio.shape[0]))

    if(i == 0):
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
        f0 = pitch_detection(AavPDS, AfreqX, numberofcompr)

        #normalise to compensate for loudness        
        AavPDSshort = AavPDSshort/max(AavPDSshort)   

        #Calculate features
        # energy1 = getEnergyInHarmonic(AavPDSshort, f0, 1,framelength, note)
        # energy2 = getEnergyInHarmonic(AavPDSshort, f0, 2,framelength, note)
        # energy3 = getEnergyInHarmonic(AavPDSshort, f0, 3,framelength, note)
        # energy4 = getEnergyInHarmonic(AavPDSshort, f0, 4,framelength, note)

    if(i == 1):
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
        f0 = pitch_detection(DavPDS, DfreqX, numberofcompr)

        #normalise to compensate for loudness        
        DavPDSshort = DavPDSshort/max(DavPDSshort)   

        #Calculate features
        # energy1 = getEnergyInHarmonic(DavPDSshort, f0, 1,framelength, note)
        # energy2 = getEnergyInHarmonic(DavPDSshort, f0, 2,framelength, note)
        # energy3 = getEnergyInHarmonic(DavPDSshort, f0, 3,framelength, note)
        # energy4 = getEnergyInHarmonic(DavPDSshort, f0, 4,framelength, note)

    if(i == 2):
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
        f0 = pitch_detection(EavPDS, EfreqX, numberofcompr)

        #normalise to compensate for loudness        
        EavPDSshort = EavPDSshort/max(EavPDSshort)   

        #Calculate features
        # energy1 = getEnergyInHarmonic(EavPDSshort, f0, 1,framelength, note)
        # energy2 = getEnergyInHarmonic(EavPDSshort, f0, 2,framelength, note)
        # energy3 = getEnergyInHarmonic(EavPDSshort, f0, 3,framelength, note)
        # energy4 = getEnergyInHarmonic(EavPDSshort, f0, 4,framelength, note)

    if(i == 3):
        standard = 1318.51
        note = 'E'  

        #Reduce the frequency resolution and take the FFT
        avPDS = get_average_pds(audio,framelength,frameskip)
        freqX = np.fft.fftfreq(len(avPDS), 1/fs)

        #Remove negative frequency parts and remove unnecessary data
        freqXshort = np.zeros(7000)
        avPDSshort = np.zeros(7000)
        for j in range(7000):
            avPDSshort[j] = avPDS[j]
            freqXshort[j] = freqX[j]

        #Find the pitch of the current audio file
        f0 = pitch_detection(avPDS, freqX, numberofcompr)

        #normalise to compensate for loudness        
        avPDSshort = avPDSshort/max(avPDSshort)   

    if(i == 4):
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
        f0 = pitch_detection(GavPDS, GfreqX, numberofcompr)

        #normalise to compensate for loudness        
        GavPDSshort = GavPDSshort/max(GavPDSshort)   

        #Calculate features
        # energy1 = getEnergyInHarmonic(GavPDSshort, f0, 1,framelength, note)
        # energy2 = getEnergyInHarmonic(GavPDSshort, f0, 2,framelength, note)
        # energy3 = getEnergyInHarmonic(GavPDSshort, f0, 3,framelength, note)
        # energy4 = getEnergyInHarmonic(GavPDSshort, f0, 4,framelength, note)
    
    #Find the pitch of the current audio file
    print("True expected frequency: " + str(standard) + " Hz")
    print("Pitch detector result: " + str(f0) + " Hz\n")
    freqArr[i] = f0

#saveToTextFile("f3freq.txt", freqArr)


fig = plt.figure(figsize=(6, 4))
t = np.arange(-5.0, 1.0, 0.1)

sub1 = fig.add_subplot(221) # instead of plt.subplot(2, 2, 1)
sub1.set_title('A4') # non OOP: plt.title('The function f')
sub1.set_xlim(0, 7000)
sub1.plot(AfreqXshort,AavPDSshort)

sub2 = fig.add_subplot(222)
sub2.set_title('D4')
sub2.set_xlim(0, 7000)
sub2.plot(DfreqXshort,DavPDSshort)

sub3 = fig.add_subplot(223)
sub3.set_title('E5')
sub3.set_xlim(0, 7000)
sub3.plot(EfreqXshort,EavPDSshort)

sub4 = fig.add_subplot(224)
sub4.set_title('G3')
sub4.set_xlim(0, 7000)
sub4.plot(GfreqXshort,GavPDSshort)

plt.tight_layout()
plt.show()


    













#Plot the FFT with reduced frequency resolution
# plt.plot(freqXshort,avPDSshort) 
# plt.title("FFT")
# plt.ylabel("Amplitude")
# plt.xlabel("Time")
# plt.xlim(0,7000)
# plt.grid()
# plt.show()






