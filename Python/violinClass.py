from msilib.schema import Class
import matplotlib.pyplot as plt
import numpy as np
from ctypes import sizeof
import os
import os.path
from scipy.io.wavfile import read


class Violin:
    # instance attributes
    def __init__(self, name, note, standard):
        self.name = name
        self.note = note
        self.standard = standard


    # instance method
    def readInData(self):
        # Get the list of all files and directories
        path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData"
        dir_list = os.listdir(path)
        return dir_list

    
    def pitch_detection(self, fftdata, freqScale, numOfComp, standard):
        f0value = standard*0.75
        yMax = 0
        compressedSum = np.zeros(int(fftdata.shape[0]))
        compressed = np.zeros(int(fftdata.shape[0]/numOfComp))
        compressedSumZoomed = np.zeros(80)
        offset = int(f0value*len(compressedSum)/96000)
        print("offset: " + str(offset))
        print("standard: " + str(standard))
        print("f0: " + str(f0value))
        
        for i in range(numOfComp):
            compressed = a1.compression(fftdata, i+1)
            compressed = np.pad(compressed, (0, int(compressedSum.shape[0]) - int(compressed.shape[0])), 'constant')
            compressedSum += compressed

        for j in range(80):
            compressedSumZoomed[j] = compressedSum[j+offset]


        index = 0
        yMax = max(compressedSumZoomed)
        print("yMax: " + str(yMax))
        for k in range(compressedSumZoomed.size):
            if(yMax == compressedSumZoomed[k]):
                index = k + offset
                
        f0value = index/len(compressedSum)*(96000)

        # for i in range(compressedSum.size):
        #     if((f0value>=0.9*standard and f0value<=1.1*standard)):
        #         if(compressedSum[i]>yMax):
        #             yMax = compressedSum[i]
        #             f0value = i/len(compressedSum)*(96000)

        # plt.xlim(0, 4000)
        # plt.title("Number of compressed graphs added: " + str(numOfComp))
        # plt.plot(freqScale,compressedSum) 
        # plt.show()
        return f0value

    def compression(self ,data, numOfComp):
        compressed = np.zeros(int(data.shape[0]/numOfComp))
        for i in range(compressed.size):
            compressed[i] = data[numOfComp*i]
        return compressed

    def getframes(self ,x ,framelength ,frameskip):
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
        #print("Number of frames: " + str(nFrames))

        # Prepare output matrix
        F = [[0 for i in range(nFrames)] for j in range(framelength)]
        F = np.reshape(F, (framelength,nFrames))

        # Now divide x into frames
        for i in range(nFrames):
            xIndex = (i)*frameskip + 1
            F[:,i] = x[xIndex:xIndex+framelength, 0]

        return F


    def get_average_pds(self ,x ,framelength ,frameskip):
        avPDS = 0
        # Divide input signal into frames
        F = a1.getframes(x,framelength,frameskip)
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

    


# instantiate the object
a1 = Violin("africa1", "G3", 196)

# call our instance methods
files = a1.readInData()

framelength = 32768 #(2^14 = 16384) or 2^15 = 32768
frameskip = 2000
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

input_data = read(path + "\\" + africa1[0])
audio = input_data[1]
print("Original audio size: " + str(audio.shape[0]))


standard = 440
note = 'A'  
print(note)

#Reduce the frequency resolution and take the FFT
AavPDS = a1.get_average_pds(audio,framelength,frameskip)
AfreqX = np.fft.fftfreq(len(AavPDS), 1/fs)

#Remove negative frequency parts and remove unnecessary data
AfreqXshort = np.zeros(7000)
AavPDSshort = np.zeros(7000)
for j in range(7000):
    AavPDSshort[j] = AavPDS[j]
    AfreqXshort[j] = AfreqX[j]



f0 = a1.pitch_detection(AavPDS, AfreqX, 15, standard)
print(f0)