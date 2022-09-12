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
#A4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\classifierInput\\A4trainNew.csv")
#D4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\classifierInput\\D4trainNew.csv")
#E5data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\classifierInput\\E5trainNew.csv")
#G3data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\classifierInput\\G3trainNew.csv")
Adagiodata = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\classifierInput\\AdagioTrain.csv")

# Splitting the dataset into the Training set and Test set
X_train = Adagiodata.iloc[:, [0,1,2, 3]].values
y_train = Adagiodata.iloc[:, 4].values 

#A4dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\classifierInput\\A4testNew.csv")
#D4dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\classifierInput\\D4testNew.csv")
#E5dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\classifierInput\\E5testNew.csv")
#G3dataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\classifierInput\\G3testNew.csv")
AdagiodataTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\classifierInput\\AdagioTest.csv")
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
ax.set_title('Adagio Confusion Matrix Accuracy = ' + str(round(accuracy, 2)) + ' %.' + '\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['africa', 'conv', 'fact']); ax.yaxis.set_ticklabels(['africa', 'conv', 'fact']);
#plt.show()


save_multi_image("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\naiveExperiment\\naiveExp3Classes\\logRegResults\\NE3CcmAdagioLR.pdf")