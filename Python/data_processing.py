from ctypes import sizeof
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile

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




#get average pds
sample_rate, samples = wavfile.read('A1-A4-001.wav')

#transform with FFT


#Normalise peaks to compensate for different loudness



#functions
#Comment Code Block Ctrl+K+C/Ctrl+K+U
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

def get_average_pds():
    # Divide input signal into frames


    # Accumulated FFTs of windowed frames


    # Normalise by number of accumulations


    # Return average spectrum
    avPDS = 0
    return avPDS



# function F = getframes(x,framelength,frameskip)



# Lx = length(x);

# if (Lx < framelength)
#  disp('ERROR: x is shorter than even a single frame');
#  return;
# end

# % Determine number of frames that can be extracted from data vector x.
# % "fix" takes integer part, like "trunc" in matlab
# nFrames = fix((Lx-framelength)/frameskip + 1);

# % Prepare output matrix
# F = zeros(framelength,nFrames);

# % Now divide x into frames
# for i = 1:nFrames
#  xIndex = (i-1)*frameskip + 1;
#  F(:,i) = x(xIndex:xIndex+framelength-1);
# end



# Divide data vector x into frames.
# F will be Nf by Lf, where NF is the number of frames and Lf the frame length
def getframes(x,framelength,frameskip):
    [r,c] = sizeof(x)
    print(r,c)

    if(r!=1 & c!=1):
        print("ERROR: x should be a vector")
        return

    # Make sure x is a column vector
    if (c > 1):
        x = x.transpose()

    Lx = x.length()


