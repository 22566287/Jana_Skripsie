import matplotlib.pyplot as plt
import numpy as np
from ctypes import sizeof
import os


def readInData():
    # Get the list of all files and directories
    path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData"
    dir_list = os.listdir(path)
    
    #print("Files and directories in '", path, "' :")
    
    # prints all files
    #print(dir_list)

    return dir_list



def pitch_detection(fftdata, freqScale, numOfComp):
    f0value = 0
    yMax = 0
    compressedSum = np.zeros(int(fftdata.shape[0]))
    compressed = np.zeros(int(fftdata.shape[0]/numOfComp))
    
    for i in range(numOfComp):
        compressed = compression(fftdata, i+1)
        compressed = np.pad(compressed, (0, int(compressedSum.shape[0]) - int(compressed.shape[0])), 'constant')
        compressedSum += compressed

    for i in range(compressedSum.size):
        if(compressedSum[i]>yMax):
            yMax = compressedSum[i]
            f0value = i/len(compressedSum)*(96000)

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

    # Return average spectrum
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
    # "fix" takes integer part, like "trunc" in matlab
    nFrames = int((Lx-framelength)/frameskip + 1)
    print("Number of frames: " + str(nFrames))

    # Prepare output matrix
    F = [[0 for i in range(nFrames)] for j in range(framelength)]
    F = np.reshape(F, (framelength,nFrames))

    # Now divide x into frames
    for i in range(nFrames):
        xIndex = (i)*frameskip + 1
        F[:,i] = x[xIndex:xIndex+framelength, 0]

    return F


def plotFFTs(x,y, xlim):
    #Plot the FFT with reduced frequency resolution
    plt.plot(x,y) 
    plt.title("FFT")
    plt.ylabel("Amplitude")
    plt.xlabel("Frequency")
    #plt.xlim(0, xlim)
    plt.grid()
    plt.show()


def saveToTextFile(fileName, data):
    file = open(fileName, "w") 
    for i in range(len(data)):
        file.write(str(data[i]) + " \n") 
    file.close() 


def getEnergyInHarmonic(x, f0, feature, framelength):
    energy = 0
    numberOfHarmonic = 1
    n = int(f0*framelength/96000)        #index of f0 in x[n]
    print(n)
    offset = int(4.5*framelength/96000)  #offset of 4Hz for 0.1 of max
    area = 2*offset                 #here n is 20

    if(feature == 1):
        while(numberOfHarmonic < 20):
            if(f0*numberOfHarmonic > 1000*1.1):
                energy = np.log(energy)
                print("Energy in feature 1: " + str(energy))
                break
            for i in range(area):
                energy += pow(x[n*numberOfHarmonic-offset+i],2)
            numberOfHarmonic += 1


    if(feature == 2):
        while(numberOfHarmonic < 20):
            if(f0*numberOfHarmonic > 2000*1.1 and f0*numberOfHarmonic < 3000*0.9):
                energy = np.log(energy)
                print("Energy in feature 2: " + str(energy))
                break
            for i in range(area):
                energy += pow(x[n*numberOfHarmonic-offset+i],2)
            numberOfHarmonic += 1


    return energy
    

