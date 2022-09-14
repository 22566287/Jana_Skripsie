# Iris dataset logistic regression example
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

# # Importing the dataset
# dataset = pd.read_csv("C:\\Users\\Jana\\seaborn-data\\iris.csv")
# dataset.describe()

# # Splitting the dataset into the Training set and Test set
# X = dataset.iloc[:, [0,1,2, 3]].values
# y = dataset.iloc[:, 4].values 
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# # Fitting Logistic Regression to the Training set
# from sklearn.linear_model import LogisticRegression
# classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
# classifier.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
# # Predict probabilities
# probs_y=classifier.predict_proba(X_test)### Print results 
# probs_y = np.round(probs_y, 2)

# from sklearn.metrics import confusion_matrix, accuracy_score
# cm = confusion_matrix(y_test, y_pred)
# print(cm)
# accuracy = accuracy_score(y_test, y_pred)*100
# print('Accuracy of the model:' + str(round(accuracy, 2)) + ' %.')

# # Plot confusion matrix
# import seaborn as sns
# import pandas as pd
# # confusion matrix sns heatmap 
# ax = plt.axes()
# df_cm = cm
# sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt='d',cmap="Blues", ax = ax )
# ax.set_title('Confusion Matrix')
# plt.show()

# Importing the dataset
#A4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\classifierInput\\A4train.csv")
#D4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\classifierInput\\D4train.csv")
#E5data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\classifierInput\\E5train.csv")
#G3data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\classifierInput\\G3train.csv")
Adagiodata = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\classifierInput\\AdagioTrain5again.csv")

# Splitting the dataset into the Training set and Test set
X_train = Adagiodata.iloc[:, [0,1,2, 3]].values
y_train = Adagiodata.iloc[:, 4].values 

#A4dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\classifierInput\\A4testNew.csv")
#D4dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\classifierInput\\D4testNew.csv")
#E5dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\classifierInput\\E5testNew.csv")
#G3dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\classifierInput\\G3testNew.csv")
AdagiodataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\classifierInput\\AdagioTest.csv")
X_test = AdagiodataTest.iloc[:, [0,1,2, 3]].values
y_test = AdagiodataTest.iloc[:, 4].values 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
classifier.fit(X_train, y_train)

# Retrieve the model parameters.
print(classifier.intercept_)   #intercepts of the three classes
print(classifier.coef_)  #weights of the four features for each class
print(classifier.coef_[0,1])


x1 = (-classifier.coef_[0,0] - classifier.coef_[0,2]*Adagiodata.iloc[0,1])/classifier.coef_[0,1]
x2 = (-classifier.coef_[0,0] - classifier.coef_[0,2]*Adagiodata.iloc[2,1])/classifier.coef_[0,1]
plt.axline((x1, Adagiodata.iloc[0,1]), (x2, Adagiodata.iloc[2,1]), color = "green")
x1 = (-classifier.coef_[1,0] - classifier.coef_[1,2]*Adagiodata.iloc[0,1])/classifier.coef_[1,1]
x2 = (-classifier.coef_[1,0] - classifier.coef_[1,2]*Adagiodata.iloc[2,1])/classifier.coef_[1,1]
plt.axline((x1, Adagiodata.iloc[0,1]), (x2, Adagiodata.iloc[2,1]), color = "red")
x1 = (-classifier.coef_[2,0] - classifier.coef_[2,2]*Adagiodata.iloc[0,1])/classifier.coef_[2,1]
x2 = (-classifier.coef_[2,0] - classifier.coef_[2,2]*Adagiodata.iloc[2,1])/classifier.coef_[2,1]
plt.axline((x1, Adagiodata.iloc[0,1]), (x2, Adagiodata.iloc[2,1]), color = "blue")


# # Plot the data and the classification with the decision boundary.
# xmin, xmax = -3, 3
# ymin, ymax = -1, 5
# xd = np.array([xmin, xmax])
# yd = m*xd + c
# plt.plot(xd, yd, 'k', lw=1, ls='--')
# plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
# plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
annotations=["A1","A2","C1","C10","C11","C12","C13","C2","C3","C4","C5","C6","C7","C8","C9","F1","F2","F3"]
annotationsTrain=["A1","A1","A1","A1","A1","A2","A2","A2","A2","A2","C1","C1","C1","C1","C1",
                "C10","C10","C10","C10","C10","C11","C11","C11","C11","C11","C12","C12","C12","C12","C12",
                "C13","C13","C13","C13","C13","C2","C2","C2","C2","C2","C3","C3","C3","C3","C3",
                "C4","C4","C4","C4","C4","C5","C5","C5","C5","C5","C6","C6","C6","C6","C6",
                "C7","C7","C7","C7","C7","C8","C8","C8","C8","C8","C9","C9","C9","C9","C9",
                "F1","F1","F1","F1","F1","F2","F2","F2","F2","F2","F3","F3","F3","F3","F3"]

plt.scatter(X_train[:,0:1], X_train[:,1:2], label='Training set')
plt.scatter(X_test[:,0:1], X_test[:,1:2], label='Test set')
plt.title("Feature 1 and 2 of Adagio logistic regression test and training set")
plt.ylabel("F4: >3kHz")
plt.xlabel("F1: <1kHz")
plt.legend()
for i, label in enumerate(annotations):
    plt.annotate(label, (X_test[i,0:1], X_test[i,1:2]))
for i, label in enumerate(annotationsTrain):
    plt.annotate(label, (X_train[i,0:1], X_train[i,1:2]))
plt.show()

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# Predict probabilities
probs_y=classifier.predict_proba(X_test)### Print results 
probs_y = np.round(probs_y, 2)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of the model:' + str(round(accuracy, 2)) + ' %.')

# Plot confusion matrix
import seaborn as sns
import pandas as pd
# confusion matrix sns heatmap 
plt.figure(figsize=(9, 6))
ax = plt.axes()
df_cm = cm
sns.heatmap(df_cm, annot=True,cmap="Blues")
ax.set_title('Adagio Confusion Matrix Accuracy = ' + str(round(accuracy, 2)) + ' %' + 'with logistic regression' + '\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['africa', 'conv', 'fact']); ax.yaxis.set_ticklabels(['africa', 'conv', 'fact']);
#plt.show()


save_multi_image("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExp3ClassFrames\\logRegResults\\NE3C4cmAdagioLR5again.pdf")