import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# read in test data
A4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\A4train.csv")
print(A4data.head(5))

# Dividing Data Into Features and Labels
print (A4data.columns)
feature_columns = (['feature1', 'feature2', 'feature3', 'feature4'])
X = A4data[feature_columns].values
y = A4data['violin'].values   #Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Splitting the Data into Training and Testing Dataset
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)#Fitting the Model and Making Predictions 
X_train = X
y_train = y
A4dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\A4test.csv")
X_test = A4data[feature_columns].values
y_test = A4data['violin'].values   #Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test = le.fit_transform(y_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

classifier = KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix        #Calculating Model Accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of the model:' + str(round(accuracy, 2)) + ' %.')
