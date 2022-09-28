import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()


# read in training data
A1train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\A1Train.csv")
A2train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\A2Train.csv")
C1train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C1Train.csv")
C10train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C10Train.csv")
C11train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C11Train.csv")
C12train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C12Train.csv")
C13train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C13Train.csv")
C2train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C2Train.csv")
C3train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C3Train.csv")
C4train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C4Train.csv")
C5train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C5Train.csv")
C6train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C6Train.csv")
C7train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C7Train.csv")
C8train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C8Train.csv")
C9train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C9Train.csv")
F1train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\F1Train.csv")
F3train = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\F3Train.csv")

# read in testing data
A1Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\A1Test.csv")
A2Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\A2Test.csv")
C1Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C1Test.csv")
C10Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C10Test.csv")
C11Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C11Test.csv")
C12Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C12Test.csv")
C13Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C13Test.csv")
C2Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C2Test.csv")
C3Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C3Test.csv")
C4Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C4Test.csv")
C5Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C5Test.csv")
C6Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C6Test.csv")
C7Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C7Test.csv")
C8Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C8Test.csv")
C9Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\C9Test.csv")
F1Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\F1Test.csv")
F3Test = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\classifierInput\\F3Test.csv")

train = [A1train,A2train,C1train,C10train,C11train,C12train,C13train,C2train,C3train,C4train,C5train,C6train,C7train,C8train,C9train,F1train,F3train]
test = [A1Test,A2Test,C1Test,C10Test,C11Test,C12Test,C13Test,C2Test,C3Test,C4Test,C5Test,C6Test,C7Test,C8Test,C9Test,F1Test,F3Test]
violin = ['A1','A2','C1','C10','C11','C12','C13','C2','C3','C4','C5','C6','C7','C8','C9','F1','F3']
#y_test_total = [0]*17
y_test_total = np.zeros(17)
y_pred_total = np.zeros(17)

for i in range(17):
    # Dividing training data into Features and Labels
    feature_columns = (['feature1', 'feature2', 'feature3', 'feature4'])
    X_train = train[i][feature_columns].values
    y_train = train[i]['violin'].values   #Label Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)

    # Dividing testing data into Features and Labels
    X_test = test[i][feature_columns].values
    y_test = test[i]['violin'].values   #Label Encoding
    if(y_test == 'africa'): y_test = 0
    if(y_test == 'conv'): y_test = 1
    if(y_test == 'fact'): y_test = 2
    y_test_total[i] = y_test

    # Fitting Logistic Regression to the Training set
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import binarize
    THRESHOLD = 0.5
    classifier = LogisticRegression(random_state = 0, solver='lbfgs', multi_class='auto')
    classifier.fit(X_train, y_train)
    y_pred = np.where(classifier.predict_proba(X_test)[:,1] > THRESHOLD, 1, 0)
    #y_pred = classifier.predict(X_test)
    print("y_test: " + str(y_test) + " ; y_pred: " + str(y_pred[0]) + " ; " + violin[i])
    y_pred_total[i] = y_pred

#Confusion Matrix
confusion_matrix = confusion_matrix(y_test_total, y_pred_total)
#confusion_matrix        #Calculating Model Accuracy
accuracy = accuracy_score(y_test_total, y_pred_total)*100
print('Accuracy of the model:' + str(round(accuracy, 2)) + ' %.')

#add violin names to cm
violin_names = ['','A1,A2\n','','','C1,C2,C3,C4,\n C5,C6,C7,C8,C9,\n C10,C11,C12,C13\n','','','F1,F3\n','']
group_counts = ["{0:0.0f}".format(value) for value in confusion_matrix.flatten()]
labels = [f"{v1}\n{v2}" for v1,v2 in zip(violin_names,group_counts)]
labels = np.asarray(labels).reshape(3,3)


plt.figure(figsize=(9, 6))
ax = sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues')
ax.set_title('Confusion Matrix Accuracy = ' + str(round(accuracy, 2)) + ' %' + ' with Logis (THRESHOLD = 0.5) for LOO experiment' + '\n\n')
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ')
ax.xaxis.set_ticklabels(['africa','conv','fact']); ax.yaxis.set_ticklabels(['africa', 'conv','fact']);
#plt.show()

save_multi_image("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\LOOexp\\LogRegresults\\LOOcmLogisDefault.pdf")