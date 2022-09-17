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
#A4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\A4train.csv")
#D4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\D4train.csv")
#E5data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\E5train.csv")
#G3data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\G3train.csv")
Adagiodata = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\AdagioTrain.csv")
# print(D4data.head(15))

# Dividing Data Into Features and Labels
#feature_columns = (['feature1', 'feature2', 'feature3', 'feature4'])
feature_columns = (['fundamental', 'harmonic2', 'harmonic3', 'harmonic4', 'harmonic5',
                    'harmonic6', 'harmonic7', 'harmonic8', 'harmonic9', 'harmonic10'])
X_train = Adagiodata[feature_columns].values
y_train = Adagiodata['violin'].values   #Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

#Splitting the Data into Training and Testing Dataset
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)#Fitting the Model and Making Predictions 


#A4dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\A4test.csv")
#D4dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\D4test.csv")
#E5dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\E5test.csv")
#G3dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\G3test.csv")
AdagiodataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\classifierInput\\AdagioTest.csv")

X_test = AdagiodataTest[feature_columns].values
y_test = AdagiodataTest['violin'].values   #Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test = le.fit_transform(y_test)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix        #Calculating Model Accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of the model:' + str(round(accuracy, 2)) + ' %.')

import seaborn as sns
plt.figure(figsize=(9, 6))
ax = sns.heatmap(confusion_matrix, annot=True, cmap='Blues')

ax.set_title('Adagio Confusion Matrix Accuracy = ' + str(round(accuracy, 2)) + ' %' + ' with kNN' + '\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['africa', 'conv', 'fact']); ax.yaxis.set_ticklabels(['africa', 'conv', 'fact']);



## Display the visualization of the Confusion Matrix.
# print(X_train[0].shape)
# print(y_train.shape)
# annotations=["A1","A1","A2","A2","C1","C1","C10","C10","C11","C11","C12","C12","C13","C13",
#         "C2","C2","C3","C3","C4","C4","C5","C5","C6","C6","C7","C7","C8","C8","C9","C9",
#         "F1","F1","F2","F2","F3","F3"]
# annotations=["A1","A2","C1","C10","C11","C12","C13",
        # "C2","C3","C4","C5","C6","C7","C8","C9",
        # "F1","F2","F3"]




# plt.scatter(X_train[2], X_train[3])
# plt.scatter(X_test[0], X_test[1])
#plt.show()
save_multi_image("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\harmonic10Features\\kNNresults\\NE10cmAdagiokNN.pdf")


# plt.figure()
# a = plt.scatter(X_train[:,0:1], X_train[:,3:4], label='Training set')
# b = plt.scatter(X_test[:,0:1], X_test[:,3:4], label='Test set')
# plt.title("Feature 1 and 4 of Adagio kNN test and training set")
# plt.ylabel("F4: >3kHz")
# plt.xlabel("F1: <1kHz")
# plt.legend()
# for i, label in enumerate(annotations):
#     plt.annotate(label, (X_test[i,0:1], X_test[i,3:4]))
# plt.show()