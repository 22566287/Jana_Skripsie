import matplotlib.pyplot as plt
import numpy as np
from ctypes import sizeof



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

    plt.xlim(0, 4000)
    plt.title("Number of compressed graphs added: " + str(numOfComp))
    plt.plot(freqScale,compressedSum) 
    plt.show()
    
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



# Divide data vector x into frames.
# F will be Nf by Lf, where NF is the number of frames and Lf the frame length
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
    # print(len(F))   #number of rows
    # print(len(F[0]))    #number of columns
    # print(F.shape)

    # Now divide x into frames
    for i in range(nFrames):
        xIndex = (i)*frameskip + 1
        F[:,i] = x[xIndex:xIndex+framelength, 0]
    #print(F[4:8])
    return F
    

