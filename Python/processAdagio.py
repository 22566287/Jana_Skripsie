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

#feat1, feat2, feat3, feat4 = [0]*90, [0]*90, [0]*90, [0]*90
# feat1, feat2, feat3, feat4, feat5 = [0]*90, [0]*90, [0]*90, [0]*90, [0]*90
# feat6, feat7, feat8, feat9, feat10 = [0]*90, [0]*90, [0]*90, [0]*90, [0]*90
feat1, feat2, feat3, feat4, feat5 = [0]*18, [0]*18, [0]*18, [0]*18, [0]*18
feat6, feat7, feat8, feat9, feat10 = [0]*18, [0]*18, [0]*18, [0]*18, [0]*18

#violin = ["africa1", "africa2", "conv1", "conv10", "conv11", "conv12", "conv13",  "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "conv9", "fact1", "fact3", "fact2"]
violin = ["africa", "africa", "conv", "conv", "conv", "conv", "conv",  "conv", "conv", "conv", "conv", "conv", "conv", "conv", "conv", "fact", "fact", "fact"]
# violin = ["africa","africa","africa","africa","africa",
#     "africa","africa","africa", "africa","africa",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "conv","conv","conv","conv","conv",
#     "fact","fact","fact","fact","fact",
#     "fact","fact","fact","fact","fact",
#     "fact","fact","fact","fact","fact"]
# violin = ["africa","africa","africa","africa",
#     "conv","conv","conv","conv",
#     "conv","conv","conv","conv",
#     "conv","conv","conv","conv",
#     "conv","conv","conv","conv",
#     "conv","conv","conv","conv",
#     "conv","conv","conv","conv",
#     "conv","conv","fact","fact",
#     "fact","fact","fact","fact"]

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
print(len(violin))


#read data names from folder
# Get the list of all files and directories
pathOfFiles = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\adagioTestSet"
files = os.listdir(pathOfFiles)
print(files)
audioArr = [0]*5
frameCounter = -1

fs = 96000
#Read data from directory with specific file
path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\adagioTestSet"
for i in range(17):
    if(files[i] != 'desktop.ini'):
        input_data = read(path + "\\" + files[i])
        audio = input_data[1]
    print(i)

    #Split training set into 4 vectors instead of one for better results in classifier
    audioArr = timeFrame(audio, audioArr,1)
    frameCounter = -1

    for m in range(1):
        frameCounter = frameCounter + 1
        #Reduce the frequency resolution and take the FFT
        AavPDS = get_average_pds(audioArr[m],framelength,frameskip)
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
        totalE1, totalE2, totalE3, totalE4 = 0, 0, 0, 0
        totalE5, totalE6, totalE7, totalE8, totalE9, totalE10 = 0, 0, 0, 0, 0, 0
        
        for k in range(4):
            # Meinel features
            # feat1[5*i + frameCounter] = getEnergyInHarmonic(AavPDSshort, f0val[k], 1,framelength, noteval[k])
            # totalE1 = totalE1 + feat1[5*i + frameCounter]

            # feat2[5*i + frameCounter] = getEnergyInHarmonic(AavPDSshort, f0val[k], 2,framelength, noteval[k])
            # totalE2 = totalE2 + feat2[5*i + frameCounter]

            # feat3[5*i + frameCounter] = getEnergyInHarmonic(AavPDSshort, f0val[k], 3,framelength, noteval[k])
            # totalE3 = totalE3 + feat3[5*i + frameCounter]

            # feat4[5*i + frameCounter] = getEnergyInHarmonic(AavPDSshort, f0val[k], 4,framelength, noteval[k])
            # totalE4 = totalE4 + feat4[5*i + frameCounter]

            # First 10 harmonic features
            feat1[1*i + frameCounter] = energyInFirstTenHarmonics(AavPDSshort, f0val[k], 1,framelength)
            totalE1 = totalE1 + feat1[1*i + frameCounter]

            feat2[1*i + frameCounter] = energyInFirstTenHarmonics(AavPDSshort, f0val[k], 2,framelength)
            totalE2 = totalE2 + feat2[1*i + frameCounter]

            feat3[1*i + frameCounter] = energyInFirstTenHarmonics(AavPDSshort, f0val[k], 3,framelength)
            totalE3 = totalE3 + feat3[1*i + frameCounter]

            feat4[1*i + frameCounter] = energyInFirstTenHarmonics(AavPDSshort, f0val[k], 4,framelength)
            totalE4 = totalE4 + feat4[1*i + frameCounter]

            feat5[1*i + frameCounter] = energyInFirstTenHarmonics(AavPDSshort, f0val[k], 5,framelength)
            totalE5 = totalE5 + feat5[1*i + frameCounter]

            feat6[1*i + frameCounter] = energyInFirstTenHarmonics(AavPDSshort, f0val[k], 6,framelength)
            totalE6 = totalE6 + feat6[1*i + frameCounter]

            feat7[1*i + frameCounter] = energyInFirstTenHarmonics(AavPDSshort, f0val[k], 7,framelength)
            totalE7 = totalE7 + feat7[1*i + frameCounter]

            feat8[1*i + frameCounter] = energyInFirstTenHarmonics(AavPDSshort, f0val[k], 8,framelength)
            totalE8 = totalE8 + feat8[1*i + frameCounter]

            feat9[1*i + frameCounter] = energyInFirstTenHarmonics(AavPDSshort, f0val[k], 9,framelength)
            totalE9 = totalE9 + feat9[1*i + frameCounter]

            feat10[1*i + frameCounter] = energyInFirstTenHarmonics(AavPDSshort, f0val[k], 10,framelength)
            totalE10 = totalE10 + feat10[1*i + frameCounter]
        
        feat1[1*i + frameCounter] = totalE1
        feat2[1*i + frameCounter] = totalE2
        feat3[1*i + frameCounter] = totalE3
        feat4[1*i + frameCounter] = totalE4
        feat5[1*i + frameCounter] = totalE5
        feat6[1*i + frameCounter] = totalE6
        feat7[1*i + frameCounter] = totalE7
        feat8[1*i + frameCounter] = totalE8
        feat9[1*i + frameCounter] = totalE9
        feat10[1*i + frameCounter] = totalE10


#saveToExcelFile('A4.xlsx', 'A4.csv', feat1A, feat2A, feat3A, feat4A)
#plotFFTs(AfreqXshort,AavPDSshort, 7000)

# Save data to excel sheet
data = pd.DataFrame({col1:feat1,col2:feat2,col3:feat3,col4:feat4,col5:feat5,col6:feat6,col7:feat7,col8:feat8,col9:feat9,col10:feat10,col11:violin})
energyPath = 'C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\'
data.to_excel(energyPath + 'AdagioTest.xlsx', sheet_name='sheet1', index=False)

read_file = pd.read_excel ('C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\AdagioTest.xlsx')
read_file.to_csv ('C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\AdagioTest.csv', index = None, header=True)





