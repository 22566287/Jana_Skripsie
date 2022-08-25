#Importing the Required Libraries and Loading the Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

A4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\A4\\A4.csv")
D4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\D4\\D4.csv")
print(A4data.head(10))

group = [1,1,2,2,2,2,2,2,2,2,2,2,3,3,3]


#Dividing Data Into Features and Labels
print (A4data.columns)
feature_columns = (['feature_1', 'feature_2', 'feature_3', 'feature_4'])
X = A4data[feature_columns].values
y = A4data['violin'].values   #Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

A4data.plot(kind ="scatter", x ='feature_1', y ='feature_2')
A4data.plot(kind ="scatter", x ='feature_3', y ='feature_4')
plt.grid()
plt.show()


sns.set_style("whitegrid")
# sepal_length, petal_length are iris feature data height used to define Height of graph whereas hue store the class of iris dataset.
sns.FacetGrid(A4data, hue ="violin", height = 6).map(plt.scatter, 'feature_1', 'feature_2').add_legend()
sns.FacetGrid(A4data, hue ="violin", height = 6).map(plt.scatter, 'feature_3', 'feature_4').add_legend()
sns.FacetGrid(D4data, hue ="violin", height = 6).map(plt.scatter, 'feature_1', 'feature_2').add_legend()
plt.show()
