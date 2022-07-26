# This file contains all the functions needed to process and store data.

import matplotlib.pyplot as plt
import numpy as np
from ctypes import sizeof
import os
import os.path
import pandas as pd


def readInData():
    # Get the list of all files and directories
    path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\trainingSet"
    dir_list = os.listdir(path)
    return dir_list



def pitch_detection(fftdata, freqScale, numOfComp, standard):
    f0value = standard*0.97
    yMax = 0
    compressedSum = np.zeros(int(fftdata.shape[0]))
    compressed = np.zeros(int(fftdata.shape[0]/numOfComp))

    loweroffset = int(f0value*len(compressedSum)/96000)
    upperoffset = int(standard*1.03*len(compressedSum)/96000)
    area = upperoffset - loweroffset
    compressedSumZoomed = np.zeros(area)
    
    #get sum of compressed graphs 
    for i in range(numOfComp):
        compressed = compression(fftdata, i+1)
        compressed = np.pad(compressed, (0, int(compressedSum.shape[0]) - int(compressed.shape[0])), 'constant')
        compressedSum += compressed

    #isolate part of sum compressed graphs around standard frequency
    for j in range(area):
        compressedSumZoomed[j] = compressedSum[j+loweroffset]

    #get yMax of the isolated part
    index = 0
    yMax = max(compressedSumZoomed)
    for k in range(compressedSumZoomed.size):
        if(yMax == compressedSumZoomed[k]):
            index = k + loweroffset
    
    #calculate f0
    f0value = index/len(compressedSum)*(96000)

    # plt.xlim(0, 4000)
    # plt.title("Number of compressed graphs added: " + str(numOfComp))
    # plt.plot(freqScale,compressedSum) 
    # plt.show()
    
    return f0value
        


def compression(data, numOfComp):
    compressed = np.zeros(int(data.shape[0]/numOfComp))
    for i in range(compressed.size):
        compressed[i] = data[numOfComp*i]
    return compressed



def get_average_pds(x,framelength,frameskip):
    avPDS = 0
    # Divide input signal into frames
    F = getframes(x,framelength,frameskip)
    [framelength,nframes] = F.shape

    # Accumulated FFTs of windowed frames
    S = np.zeros(framelength)
    w = np.hamming(framelength)
    for i in range(nframes):
        S = S + pow(abs(np.fft.fft(F[:,i]*w)), 2)

    # Normalise by number of accumulations
    S = S/nframes
    avPDS = S
    return avPDS



def getframes(x,framelength,frameskip):
    x = np.reshape(x, (x.shape[0],1))
    [r,c] = x.shape

    if(r!=1 and c!=1):
        print("ERROR: x should be a vector")
        return

    # Make sure x is a column vector
    if (c > 1):
        x = x.transpose()

    Lx = len(x)

    if(Lx < framelength):
        print("ERROR: x is shorter than even a single frame")
        return

    # Determine number of frames that can be extracted from data vector x.
    nFrames = int((Lx-framelength)/frameskip + 1)

    # Prepare output matrix
    F = [[0 for i in range(nFrames)] for j in range(framelength)]
    F = np.reshape(F, (framelength,nFrames))
    #print(np.shape(F))

    # Now divide x into frames
    for i in range(nFrames):
        xIndex = (i)*frameskip + 1
        F[:,i] = x[xIndex:xIndex+framelength, 0]

    return F


def timeFrame(audio, audioArr,split):
    audio = np.reshape(audio, (audio.shape[0],1))
    print(audio.shape)
    audioArr = np.array_split(audio,split)
    print(audioArr[0].shape)
    return audioArr



def plotFFTs(x,y, xlim):
    plt.plot(x,y) 
    plt.title("FFT")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    plt.xlim(0, xlim)
    plt.grid()
    plt.show()

def plot4FFTs(freqXshort, AavPDSshort, DavPDSshort, EavPDSshort, GavPDSshort):
    fig = plt.figure(figsize=(6, 4))
    sub1 = fig.add_subplot(221) # instead of plt.subplot(2, 2, 1)
    sub1.set_title('A4') # non OOP: plt.title('The function f')
    sub1.set_xlim(0, 7000)
    sub1.plot(freqXshort,AavPDSshort)
    sub2 = fig.add_subplot(222)
    sub2.set_title('D4')
    sub2.set_xlim(0, 7000)
    sub2.plot(freqXshort,DavPDSshort)
    sub3 = fig.add_subplot(223)
    sub3.set_title('E5')
    sub3.set_xlim(0, 7000)
    sub3.plot(freqXshort,EavPDSshort)
    sub4 = fig.add_subplot(224)
    sub4.set_title('G3')
    sub4.set_xlim(0, 7000)
    sub4.plot(freqXshort,GavPDSshort)
    plt.tight_layout()
    plt.show()



def saveToTextFile(fileName, data):
    save_path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\results"
    completeName = os.path.join(save_path, fileName)         

    file = open(completeName, "w") 
    for i in range(len(data)):
        file.write(str(data[i]) + " \n") 
    file.close() 

def saveToExcelFile(filenamexlsx, filenamecsv, feat1, feat2, feat3,feat4,feat5,feat6,feat7,feat8,feat9,feat10):
    #violin = ["africa1", "africa2", "conv1", "conv10",  "conv11", "conv12", "conv13", "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "conv9", "fact1", "fact2", "fact3"]
    # violin = ["africa","africa","africa","africa","africa","africa","africa", "africa", 
    # "conv","conv","conv","conv", "conv","conv","conv","conv", "conv","conv","conv","conv",
    # "conv","conv","conv","conv", "conv","conv","conv","conv", "conv","conv","conv","conv", 
    # "conv","conv","conv","conv", "conv","conv","conv","conv", "conv","conv","conv","conv", 
    # "conv","conv","conv","conv", "conv","conv","conv","conv", "conv","conv","conv","conv", 
    # "conv","conv","conv","conv","fact","fact","fact","fact", "fact","fact","fact","fact", 
    # "fact","fact","fact","fact"]
    violin = ["africa","africa","conv","conv","conv","conv", "conv","conv","conv","conv", "conv","conv","conv","conv","conv","fact","fact","fact"]
    col1 = "fundamental"
    col2 = "harmonic2"
    col3 = "harmonic3"
    col4 = "harmonic4"
    col5 = "harmonic5"
    col6 = "harmonic6"
    col7 = "harmonic7"
    col8 = "harmonic8"
    col9 = "harmonic9"
    col10 = "harmonic10"
    col11 = "violin"

    data = pd.DataFrame({col1:feat1,col2:feat2,col3:feat3,col4:feat4,col5:feat5,col6:feat6,col7:feat7,col8:feat8,col9:feat9,col10:feat10,col11:violin})
    energyPath = 'C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\'
    data.to_excel(energyPath + filenamexlsx, sheet_name='sheet1', index=False)

    read_file = pd.read_excel ('C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\' + filenamexlsx)
    read_file.to_csv ('C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\' + filenamecsv, index = None, header=True)



def getEnergyInHarmonic(x, f0, feature, framelength, note):
    energy = 0
    numberOfHarmonic = 1
    n = int(f0*framelength/96000)        #index of f0 in x[n]
    offset = int(4.5*framelength/96000)  #offset of 4Hz for 0.1 of max
    area = 2*offset                 #here n is 20
    

    if(feature == 1):       #frequencies less than 1kHz
        while(numberOfHarmonic < 20):
            if(f0*numberOfHarmonic > 1000*1.1):
                #energy = np.log(energy)
                #print("Energy in feature 1: " + str(energy))
                break
            for i in range(area):
                energy += pow(x[n*numberOfHarmonic-offset+i],2)
            numberOfHarmonic += 1


    if(feature == 2):       #frequencies between 1kHz and 2kHz
        if(note == 'G'):
            numberOfHarmonic = 6
        if(note == 'D'):
            numberOfHarmonic = 4
        if(note == 'A'):
            numberOfHarmonic = 3
        if(note == 'E'):
            numberOfHarmonic = 2

        while(numberOfHarmonic < 30):
            if(f0*numberOfHarmonic > 2000*1.1):
                #energy = np.log(energy)
                #print("Energy in feature 2: " + str(energy))
                break
            for i in range(area):
                energy += pow(x[n*numberOfHarmonic-offset+i],2)
            numberOfHarmonic += 1

    if(feature == 3):       #frequencies between 2kHz and 3kHz
        if(note == 'G'):
            numberOfHarmonic = 11
        if(note == 'D'):
            numberOfHarmonic = 7
        if(note == 'A'):
            numberOfHarmonic = 5
        if(note == 'E'):
            numberOfHarmonic = 3

        while(numberOfHarmonic < 50):
            if(f0*numberOfHarmonic > 3000*1.1):
                #energy = np.log(energy)
                #print("Energy in feature 3: " + str(energy))
                break
            for i in range(area):
                energy += pow(x[n*numberOfHarmonic-offset+i],2)
            numberOfHarmonic += 1

    if(feature == 4):       #frequencies greater than 3kHz
        if(note == 'G'):
            numberOfHarmonic = 16
        if(note == 'D'):
            numberOfHarmonic = 10
        if(note == 'A'):
            numberOfHarmonic = 7
        if(note == 'E'):
            numberOfHarmonic = 5

        while(numberOfHarmonic < 100):
            if(f0*numberOfHarmonic > 14000*1.1):
                #energy = np.log(energy)
                #print("Energy in feature 4: " + str(energy))
                break
            for i in range(area):
                energy += pow(x[n*numberOfHarmonic-offset+i],2)
            numberOfHarmonic += 1


    return energy
    
def energyInFirstTenHarmonics(x, f0, numberOfHarmonic, framelength):
    energy = 0
    n = int(f0*framelength/96000)        #index of f0 in x[n]
    offset = int(4.5*framelength/96000)  #offset of 4Hz for 0.1 of max
    area = 2*offset                 #here n is 20
    
    for i in range(area):
        energy += pow(x[n*numberOfHarmonic-offset+i],2)

    return energy
