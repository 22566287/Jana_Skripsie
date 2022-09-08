import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

# read in test data
#A4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\A4train.csv")
#D4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\D4train.csv")
#E5data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\E5train.csv")
G3data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\G3train.csv")
# print(D4data.head(15))

# Dividing Data Into Features and Labels
feature_columns = (['feature1', 'feature2', 'feature3', 'feature4'])
X = G3data[feature_columns].values
y = G3data['violin'].values   #Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Splitting the Data into Training and Testing Dataset
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)#Fitting the Model and Making Predictions 
X_train = X
y_train = y

#A4dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\A4test.csv")
#D4dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\D4test.csv")
#E5dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\E5test.csv")
G3dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierInput\\G3test.csv")

X_test = G3dataTest[feature_columns].values
y_test = G3dataTest['violin'].values   #Label Encoding
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

import seaborn as sns
plt.figure()
ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')

ax.set_title('Seaborn Confusion Matrix with labels\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')



## Display the visualization of the Confusion Matrix.
# print(X_train[0].shape)
# print(y_train.shape)

# plt.figure()
# plt.scatter(X_train[0], X_train[1])
# plt.scatter(X_train[2], X_train[3])
# plt.scatter(X_test[0], X_test[1])
#plt.show()
save_multi_image("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\classifierResults\\NEcmG3.pdf")


