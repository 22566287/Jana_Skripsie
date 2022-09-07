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

A4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\A4\\A4oud.csv")
D4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\D4\\D4.csv")
print(A4data.head(15))
print(D4data.head(15))
Adagiodata = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\Adagio.csv")

group = [1,1,2,2,2,2,2,2,2,2,2,2,3,3]


# #Dividing Data Into Features and Labels
# print (A4data.columns)
# feature_columns = (['feature_1', 'feature_2', 'feature_3', 'feature_4'])
# X = A4data[feature_columns].values
# y = A4data['violin'].values   #Label Encoding
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)

# A4data.plot(kind ="scatter", x ='feature_1', y ='feature_2')
# A4data.plot(kind ="scatter", x ='feature_3', y ='feature_4')
# plt.grid()
# plt.show()


# sns.set_style("whitegrid")
# sns.FacetGrid(A4data, hue ="violin", height = 6).map(plt.scatter, 'feature_1', 'feature_2').add_legend()
# sns.FacetGrid(A4data, hue ="violin", height = 6).map(plt.scatter, 'feature_3', 'feature_4').add_legend()
# sns.FacetGrid(D4data, hue ="violin", height = 6).map(plt.scatter, 'feature_1', 'feature_2').add_legend()
# plt.show()

print(np.shape(A4data))
print(A4data.iloc[:,0:2])

#create colors for labels
#colors = np.array(["red", "red", "green", "green", "green","green","green","green","green","green","green","green", "blue", "blue", "blue"])
colors = np.array(["red", "hotpink", "aqua", "turquoise", "lightseagreen","paleturquoise","gold","goldenrod","orange","darkorange","olive","cyan", "lime", "lawngreen", "lightgreen"])
annotations=["A1","A2","C1","C10","C2","C3","C4","C5","C6","C7","C8","C9","F1","F2","F3"]

a1 = plt.figure(1)
plt.scatter(A4data.iloc[:,0:1], A4data.iloc[:,1:2], c=colors)
plt.title("Feature 1 and 2 of A4 energy")
plt.ylabel("Feature 2")
plt.xlabel("Feature 1")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (A4data.iat[i,0], A4data.iat[i,1]))


a2 = plt.figure(2)
plt.scatter(A4data.iloc[:,2:3], A4data.iloc[:,3:4], c=colors)
plt.title("Feature 3 and 4 of A4 energy")
plt.ylabel("Feature 4")
plt.xlabel("Feature 3")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (A4data.iat[i,2], A4data.iat[i,3]))


# d1 = plt.figure(3)
# plt.scatter(D4data.iloc[:,0:1], D4data.iloc[:,1:2], c=colors)
# plt.title("Feature 1 and 2 of D4 energy")
# plt.ylabel("Feature 2")
# plt.xlabel("Feature 1")
# plt.grid()

# d2 = plt.figure(4)
# plt.scatter(D4data.iloc[:,2:3], D4data.iloc[:,3:4], c=colors)
# plt.title("Feature 3 and 4 of D4 energy")
# plt.ylabel("Feature 4")
# plt.xlabel("Feature 3")
# plt.grid()



adagio = plt.figure(5)
plt.scatter(Adagiodata.iloc[:,0:1], Adagiodata.iloc[:,1:2], c = colors)
plt.title("Feature 1 and 2 of Adagio energy")
plt.ylabel("Feature 2")
plt.xlabel("Feature 1")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (Adagiodata.iat[i,0], Adagiodata.iat[i,1]))

adagio = plt.figure(6)
plt.scatter(Adagiodata.iloc[:,2:3], Adagiodata.iloc[:,3:4], c = colors)
plt.title("Feature 3 and 4 of Adagio energy")
plt.ylabel("Feature 4")
plt.xlabel("Feature 3")
plt.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (Adagiodata.iat[i,2], Adagiodata.iat[i,3]))


#plt.show()
save_multi_image("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\results\\energyResults.pdf")


# plt.rcParams["figure.figsize"] = [7.00, 3.50]
# plt.rcParams["figure.autolayout"] = True

# fig1 = plt.figure()
# plt.plot([2, 1, 7, 1, 2], color='red', lw=5)

# fig2 = plt.figure()
# plt.plot([3, 5, 1, 5, 3], color='green', lw=5)




