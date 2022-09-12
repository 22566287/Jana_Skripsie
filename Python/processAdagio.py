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

feat1 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat2 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat3 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
feat4 = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

#violin = ["africa1", "africa2", "conv1", "conv10", "conv11", "conv12", "conv13",  "conv2", "conv3", "conv4", "conv5", "conv6", "conv7", "conv8", "conv9", "fact1", "fact3", "fact2"]
violin = ["africa", "africa", "conv", "conv", "conv", "conv", "conv",  "conv", "conv", "conv", "conv", "conv", "conv", "conv", "conv", "fact", "fact", "fact"]
col1 = "feature1"
col2 = "feature2"
col3 = "feature3"
col4 = "feature4"
col5 = "violin"


#read data names from folder
# Get the list of all files and directories
pathOfFiles = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\adagioTestSet"
files = os.listdir(pathOfFiles)
print(files)


fs = 96000
#Read data from directory with specific file
path = "C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\adagioTestSet"
for i in range(17):
    if(files[i] != 'desktop.ini'):
        input_data = read(path + "\\" + files[i])
        audio = input_data[1]
    print(i)

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
        feat1[i] = getEnergyInHarmonic(AavPDSshort, f0val[k], 1,framelength, noteval[k])
        totalE1 = totalE1 + feat1[i]

        feat2[i] = getEnergyInHarmonic(AavPDSshort, f0val[k], 2,framelength, noteval[k])
        totalE2 = totalE2 + feat2[i]

        feat3[i] = getEnergyInHarmonic(AavPDSshort, f0val[k], 3,framelength, noteval[k])
        totalE3 = totalE3 + feat3[i]

        feat4[i] = getEnergyInHarmonic(AavPDSshort, f0val[k], 4,framelength, noteval[k])
        totalE4 = totalE4 + feat4[i]
    
    feat1[i] = totalE1
    feat2[i] = totalE2
    feat3[i] = totalE3
    feat4[i] = totalE4


#saveToExcelFile('A4.xlsx', 'A4.csv', feat1A, feat2A, feat3A, feat4A)
#plotFFTs(AfreqXshort,AavPDSshort, 7000)

# Save data to excel sheet
data = pd.DataFrame({col1:feat1,col2:feat2,col3:feat3,col4:feat4,col5:violin})
energyPath = 'C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\'
data.to_excel(energyPath + 'AdagioTest.xlsx', sheet_name='sheet1', index=False)

read_file = pd.read_excel ('C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\AdagioTest.xlsx')
read_file.to_csv ('C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\AdagioTest.csv', index = None, header=True)





