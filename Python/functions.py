import matplotlib.pyplot as plt
import numpy as np
from ctypes import sizeof

def pitch_detection(fftdata, freqScale, numOfComp):
    #Idea: compute graphs and store in 2D array - do recursively
    #      Add first N points of all graphs to one graph
    #      N should be greater than potential f0
    #      find max amplitude of resultant graph - corrisponding
    #      frequency is f0
    #      return f0
    for i in range(numOfComp):
        compression(fftdata,freqScale, i+2)  #do it with recursion
        print(i+2)



def compression(data,freqS, numOfComp):
    compressed = np.zeros(int(data.shape[0]/numOfComp))
    freqX = np.fft.fftfreq(len(compressed), 1/96000*numOfComp)
    yMax = 0
    f0value = 0
    maxIndex = 0
    for i in range(compressed.size):
        compressed[i] = data[numOfComp*i]
        if(compressed[i] > yMax):
            yMax = compressed[i]
            maxIndex = i
            f0value = maxIndex/len(compressed)*(96000)
    print("max index: ")
    print(f0value)
    
    plt.xlim(0, 7000)
    plt.title(numOfComp)
    plt.plot(freqX,compressed) #plot freqX vs Hx
    plt.show()
    print(compressed.size)

def plot_a_graph():
    # x axis values 
    x = [1,2,3] 
    # corresponding y axis values 
    y = [2,4,1] 
    
    # plotting the points  
    plt.plot(x, y) 
    plt.xlabel('x - axis') 
    # naming the y axis 
    plt.ylabel('y - axis') 
    
    # giving a title to my graph 
    plt.title('My first graph!') 
    
    # function to show the plot 
    plt.show() 

#functions

# % Determine the average power density spectrum of a signal.
# %
# % Usage: y = get_average_pds(x,framelength,frameskip)
# %
# % Systems & Signals 414, Thomas Niesler, Stellenbosch University 2007-2020.

# function y = get_average_pds(x,framelength,frameskip)

# % Divide input signal into frames
# F = getframes(x,framelength,frameskip);
# [framelength,nframes] = size(F);

# %disp(['Found ' num2str(nframes) ' frames.'])

# % Accumulated FFTs of windowed frames
# S = zeros(framelength,1);
# w = hamming(framelength);
# for i = 1:nframes
#  S = S + abs(fft(F(:,i).*w)).^2;
# end

# % Normalise by number of accumulations
# S = S / nframes;

# % Return average spectrum
# y = S;

def get_average_pds(x,framelength,frameskip):
    print("hello")
    # Divide input signal into frames
    F = getframes(x,framelength,frameskip)
    print(F)

    # Accumulated FFTs of windowed frames


    # Normalise by number of accumulations


    # Return average spectrum
    # avPDS = 0
    # return avPDS



# Divide data vector x into frames.
# F will be Nf by Lf, where NF is the number of frames and Lf the frame length
def getframes(x,framelength,frameskip):
    # [r,c] = x.size
    r = len(x)
    #c = len(x[0])
    #print(x.shape)

    # if(r!=1 and c!=1):
    #     print("ERROR: x should be a vector")
    #     return

    # # Make sure x is a column vector
    # if (c > 1):
    #     x = x.transpose()

    Lx = len(x)

    if(Lx < framelength):
        print("ERROR: x is shorter than even a single frame")
        return

    # Determine number of frames that can be extracted from data vector x.
    # "fix" takes integer part, like "trunc" in matlab
    nFrames = int((Lx-framelength)/frameskip + 1)

    # Prepare output matrix
    F = np.zeros(nFrames)

    # Now divide x into frames
    for i in range(nFrames):
        xIndex = (i-1)*frameskip + 1
        F[i] = x[xIndex]
        #print(F[i])
    return F
    

