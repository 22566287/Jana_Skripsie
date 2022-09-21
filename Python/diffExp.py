from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
import pandas as pd
from sklearn.naive_bayes import GaussianNB




AdagioTrain = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\DiffExp\\classifierInput\\DiffTrain.csv")
feature_columns = (['feature1', 'feature2', 'feature3', 'feature4'])
X_train = AdagioTrain[feature_columns].values
y_train = AdagioTrain['violin'].values   #Label Encoding
print(y_train)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)
print(y_train)

AdagioTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\DiffExp\\classifierInput\\DiffTest.csv")
X_test = AdagioTest[feature_columns].values
y_test = AdagioTest['violin'].values   #Label Encoding
print(y_test)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test = le.fit_transform(y_test)
print(y_test)

print(AdagioTrain.describe())
print(AdagioTest.describe())

print(AdagioTest['feature1'].std())

gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test) 
accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)


cm = confusion_matrix(y_test, Y_pred)
accuracy = accuracy_score(y_test,Y_pred)
precision =precision_score(y_test, Y_pred,average='micro')
recall =  recall_score(y_test, Y_pred,average='micro')
f1 = f1_score(y_test,Y_pred,average='micro')
print('Confusion matrix for Naive Bayes\n',cm)
print('accuracy_Naive Bayes: %.3f' %accuracy)
print('precision_Naive Bayes: %.3f' %precision)
print('recall_Naive Bayes: %.3f' %recall)
print('f1-score_Naive Bayes : %.3f' %f1)

accuracy = accuracy*100
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(9, 6))
ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Gussian outlier detector Confusion Matrix Accuracy = ' + str(round(accuracy, 2)) + ' %' + '\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['africa','conv','fact']); ax.yaxis.set_ticklabels(['africa', 'conv','fact']);

annotationsTest=["A1","A1","A1","A1","A1","A1","A2","A2","A2","A2","A2","A2","C1","C1","C1","C1","C1",
                "C1","C2","C2","C2","C2","C2","C2","C3","C3","C3","C3","C3","C3","C8","C8","C8","C8",
                "C8","C8","F1","F1","F1","F1","F1","F1"]

fig, ax = plt.subplots(figsize=(12,6))
ax.scatter(AdagioTrain['feature1'], AdagioTrain['feature2'],label='Training set')
ax.scatter(AdagioTest['feature1'], AdagioTest['feature2'],label='Test set')
plt.legend()
ax.set_xlabel('0-1kHz')
ax.set_ylabel('1.5kHz')
for i, label in enumerate(annotationsTest):
    plt.annotate(label, (AdagioTest['feature1'][i], AdagioTest['feature2'][i]))






import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
# Plot between -10 and 10 with .001 steps.
x_axis = np.arange(-1, 4, 0.01)
# Calculating mean and standard deviation
meanTest = AdagioTest['feature1'].mean()
sdTest = AdagioTest['feature1'].std()
plt.plot(x_axis, norm.pdf(x_axis, meanTest, sdTest))

meanTrain = AdagioTrain['feature1'].mean()
sdTrain = AdagioTrain['feature1'].std()
plt.plot(x_axis, norm.pdf(x_axis, meanTrain, sdTrain))


# plt.figure(2)
# ax.scatter(AdagioTrain['feature3'], AdagioTrain['feature4'],label='Training set')
# ax.scatter(AdagioTest['feature3'], AdagioTest['feature4'],label='Test set')
# plt.legend()
# ax.set_xlabel('2-3kHz')
# ax.set_ylabel('>3kHz')
# for i, label in enumerate(annotationsTest):
#     plt.annotate(label, (AdagioTest['feature3'][i], AdagioTest['feature4'][i]))

plt.show()

