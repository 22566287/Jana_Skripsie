#Importing the Required Libraries and Loading the Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

A4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\A4.csv")
D4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\D4.csv")
E5data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\E5.csv")
G3data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\G3.csv")
Adagiodata = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\Adagio.csv")
print(A4data.head(15))


#create colors for labels
colors = np.array(["red", "red", "green", "green", "green","green","green","green","green","green","green","green", "blue", "blue", "blue"])
#colors = np.array(["red", "hotpink", "aqua", "turquoise", "lightseagreen","paleturquoise","gold","goldenrod","orange","darkorange","olive","cyan", "lime", "lawngreen", "lightgreen"])
annotations=["A1","A2","C1","C10","C2","C3","C4","C5","C6","C7","C8","C9","F1","F2","F3"]

a1 = plt.figure(1)
plt.scatter(A4data.iloc[:,0:1], A4data.iloc[:,1:2], c=colors)
plt.title("Feature 1 and 2 of A4 energy")
plt.ylabel("F2: 1kHz - 2kHz")
plt.xlabel("F1: <1kHz")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (A4data.iat[i,0], A4data.iat[i,1]))


a2 = plt.figure(2)
plt.scatter(A4data.iloc[:,2:3], A4data.iloc[:,3:4], c=colors)
plt.title("Feature 3 and 4 of A4 energy")
plt.ylabel("F4: >3kHz")
plt.xlabel("F3: 2kHz - 3kHz")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (A4data.iat[i,2], A4data.iat[i,3]))


d1 = plt.figure(3)
plt.scatter(D4data.iloc[:,0:1], D4data.iloc[:,1:2], c=colors)
plt.title("Feature 1 and 2 of D4 energy")
plt.ylabel("F2: 1kHz - 2kHz")
plt.xlabel("F1: <1kHz")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (D4data.iat[i,0], D4data.iat[i,1]))

d2 = plt.figure(4)
plt.scatter(D4data.iloc[:,2:3], D4data.iloc[:,3:4], c=colors)
plt.title("Feature 3 and 4 of D4 energy")
plt.ylabel("F4: >3kHz")
plt.xlabel("F3: 2kHz - 3kHz")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (D4data.iat[i,2], D4data.iat[i,3]))

e1 = plt.figure(5)
plt.scatter(E5data.iloc[:,0:1], E5data.iloc[:,1:2], c=colors)
plt.title("Feature 1 and 2 of E5 energy")
plt.ylabel("F2: 1kHz - 2kHz")
plt.xlabel("F1: <1kHz")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (E5data.iat[i,0], E5data.iat[i,1]))

e2 = plt.figure(6)
plt.scatter(E5data.iloc[:,2:3], E5data.iloc[:,3:4], c=colors)
plt.title("Feature 3 and 4 of E5 energy")
plt.ylabel("F4: >3kHz")
plt.xlabel("F3: 2kHz - 3kHz")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (E5data.iat[i,2], E5data.iat[i,3]))

g1 = plt.figure(7)
plt.scatter(G3data.iloc[:,0:1], G3data.iloc[:,1:2], c=colors)
plt.title("Feature 1 and 2 of G3 energy")
plt.ylabel("F2: 1kHz - 2kHz")
plt.xlabel("F1: <1kHz")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (G3data.iat[i,0], G3data.iat[i,1]))

g2 = plt.figure(8)
plt.scatter(G3data.iloc[:,2:3], G3data.iloc[:,3:4], c=colors)
plt.title("Feature 3 and 4 of G3 energy")
plt.ylabel("F4: >3kHz")
plt.xlabel("F3: 2kHz - 3kHz")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (G3data.iat[i,2], G3data.iat[i,3]))


adagio1 = plt.figure(9)
plt.scatter(Adagiodata.iloc[:,0:1], Adagiodata.iloc[:,1:2], c = colors)
plt.title("Feature 1 and 2 of Adagio energy")
plt.ylabel("F2: 1kHz - 2kHz")
plt.xlabel("F1: <1kHz")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (Adagiodata.iat[i,0], Adagiodata.iat[i,1]))

adagio2 = plt.figure(10)
plt.scatter(Adagiodata.iloc[:,2:3], Adagiodata.iloc[:,3:4], c = colors)
plt.title("Feature 3 and 4 of Adagio energy")
plt.ylabel("F4: >3kHz")
plt.xlabel("F3: 2kHz - 3kHz")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (Adagiodata.iat[i,2], Adagiodata.iat[i,3]))


#plt.show()
save_multi_image("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\results\\energyResults.pdf")





