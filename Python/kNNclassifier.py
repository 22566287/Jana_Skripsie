#Importing the Required Libraries and Loading the Dataset
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_csv("C:\\Users\\Jana\\anaconda3\\Lib\\site-packages\\bokeh\\sampledata\\_data\\iris.csv")
print(dataset.head(10))

#Dividing Data Into Features and Labels
print (dataset.columns)
feature_columns = (['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
X = dataset[feature_columns].values
y = dataset['species'].values   #Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)



#Splitting the Data into Training and Testing Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)#Fitting the Model and Making Predictions 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Confusion Matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
confusion_matrix        #Calculating Model Accuracy
accuracy = accuracy_score(y_test, y_pred)*100
print('Accuracy of the model:' + str(round(accuracy, 2)) + ' %.')

# dataset.plot(kind ="scatter", x ='sepal_length', y ='petal_length')
# plt.grid()
# plt.show()

iris = sns.load_dataset('iris')
sns.set_style("whitegrid")
 
# sepal_length, petal_length are iris feature data height used to define Height of graph whereas hue store the class of iris dataset.
plt.figure()
sns.FacetGrid(iris, hue ="species", height = 6).map(plt.scatter, 'sepal_length', 'petal_length').add_legend()
plt.show()