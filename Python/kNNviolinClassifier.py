#Importing the Required Libraries and Loading the Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

A4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\A4\\A4.csv")
D4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\D4\\D4.csv")
print(A4data.head(15))
print(D4data.head(15))

group = [1,1,2,2,2,2,2,2,2,2,2,2,3,3,3]


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

a1 = plt.figure(1)
plt.scatter(A4data.iloc[:,0:1], A4data.iloc[:,1:2], c=colors)
plt.title("Feature 1 and 2 of A4 energy")
plt.ylabel("Feature 2")
plt.xlabel("Feature 1")
plt.grid()

a2 = plt.figure(2)
plt.scatter(A4data.iloc[:,2:3], A4data.iloc[:,3:4], c=colors)
plt.title("Feature 3 and 4 of A4 energy")
plt.ylabel("Feature 4")
plt.xlabel("Feature 3")
plt.grid()


d1 = plt.figure(3)
plt.scatter(D4data.iloc[:,0:1], D4data.iloc[:,1:2], c=colors)
plt.title("Feature 1 and 2 of D4 energy")
plt.ylabel("Feature 2")
plt.xlabel("Feature 1")
plt.grid()

d2 = plt.figure(4)
plt.scatter(D4data.iloc[:,2:3], D4data.iloc[:,3:4], c=colors)
plt.title("Feature 3 and 4 of D4 energy")
plt.ylabel("Feature 4")
plt.xlabel("Feature 3")
plt.grid()


plt.show()