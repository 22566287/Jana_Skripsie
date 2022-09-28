from cmath import sqrt
from sklearn.metrics import accuracy_score,precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score ,precision_score,recall_score,f1_score
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import numpy as np
import matplotlib.pyplot as plt


AdagioTrain = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\DiffExp\\classifierInput\\Train.csv")
feature_columns = (['feature1', 'feature2', 'feature3', 'feature4'])
X_train = AdagioTrain[feature_columns].values
y_train = AdagioTrain['violin'].values   #Label Encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

AdagioTest = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\DiffExp\\classifierInput\\TestFull.csv")
X_test = AdagioTest[feature_columns].values
y_test = AdagioTest['violin'].values   #Label Encoding
print(y_test)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test = le.fit_transform(y_test)


print(AdagioTrain.describe())
print(AdagioTest.std())
print(AdagioTest.mean())

#multivariate gaussian 
mean = np.array([[AdagioTrain['feature1'].mean(),AdagioTrain['feature2'].mean(),AdagioTrain['feature3'].mean(),AdagioTrain['feature4'].mean()]])
mean = mean.transpose()
print(mean)
cov = np.array(AdagioTrain.cov())
print(mean.shape)
print(cov.shape)
X_test = X_test.transpose()
print(X_test.shape)
prob_density = [0]*6
mahDist = [0]*6

for i in range(6):
    prob_density[i] = pow(pow(np.pi*2,4)*np.linalg.det(cov),-0.5) * np.exp(-0.5*np.dot(np.dot((X_test[:,i]-mean[:,0]).transpose(),np.linalg.inv(cov)), (X_test[:,i]-mean[:,0])))
    mahDist[i] = pow(np.dot(np.dot((X_test[:,i]-mean[:,0]).transpose(),np.linalg.inv(cov)), (X_test[:,i]-mean[:,0])),0.5)
print(str(prob_density))
scale = pow(pow(np.pi*2,4)*np.linalg.det(cov),-0.5)
print(scale)
print(mahDist)

x_value = [mahDist[3],mahDist[4],mahDist[5],mahDist[0],mahDist[2],mahDist[1]]
prob = [prob_density[3]/scale,prob_density[4]/scale,prob_density[5]/scale,prob_density[0]/scale,prob_density[2]/scale,prob_density[1]/scale]

annotations=["C2","C3","F1","A1","C8","A2"]

plt.scatter(x_value,prob)
plt.plot(x_value,prob)
plt.axvline(x = 1.3, color = 'r', label = 'axvline - full height')
# plt.scatter(X_train[:,0:1], X_train[:,1:2], label='Training set')
# plt.scatter(X_test[:,0:1], X_test[:,1:2], label='Test set')
plt.title("Mahalanobis distance againt probability")
plt.ylabel("Probability")
plt.xlabel("Mahalanobis distance")
#plt.legend()
for i, label in enumerate(annotations):
    plt.annotate(label, (x_value[i], prob[i]))
plt.show()


# # feature1
# mean1 = AdagioTrain['feature1'].mean()
# sd1 = AdagioTrain['feature1'].std()
# scale = 1/(sqrt(2*np.pi)*sd1)
# print(scale)        #max value to scale density with
# prob_density1 = 1/((np.sqrt(np.pi*2)*sd1)) * np.exp(-0.5*((X_test[:,0]-mean1)/sd1)**2)
# print(prob_density1)

# # feature2
# mean2 = AdagioTrain['feature2'].mean()
# sd2 = AdagioTrain['feature2'].std()
# prob_density2 = 1/((np.sqrt(np.pi*2)*sd2)) * np.exp(-0.5*((X_test[:,0]-mean2)/sd2)**2)
# print(prob_density2)

# # feature3
# mean3 = AdagioTrain['feature3'].mean()
# sd3 = AdagioTrain['feature3'].std()
# prob_density3 = 1/((np.sqrt(np.pi*2)*sd3)) * np.exp(-0.5*((X_test[:,0]-mean3)/sd3)**2)
# print(prob_density3)

# # feature4
# mean4 = AdagioTrain['feature4'].mean()
# sd4 = AdagioTrain['feature4'].std()
# prob_density4 = 1/((np.sqrt(np.pi*2)*sd4)) * np.exp(-0.5*((X_test[:,0]-mean4)/sd4)**2)
# print(prob_density4)



# gaussian = GaussianNB()
# gaussian.fit(X_train, y_train)
# Y_pred = gaussian.predict(X_test) 
# accuracy_nb=round(accuracy_score(y_test,Y_pred)* 100, 2)
# acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)

# cm = confusion_matrix(y_test, Y_pred)
# accuracy = accuracy_score(y_test,Y_pred)
# precision =precision_score(y_test, Y_pred,average='micro')
# recall =  recall_score(y_test, Y_pred,average='micro')
# f1 = f1_score(y_test,Y_pred,average='micro')
# print('Confusion matrix for Naive Bayes\n',cm)
# print('accuracy_Naive Bayes: %.3f' %accuracy)
# print('precision_Naive Bayes: %.3f' %precision)
# print('recall_Naive Bayes: %.3f' %recall)
# print('f1-score_Naive Bayes : %.3f' %f1)

# accuracy = accuracy*100
# import seaborn as sns
# import matplotlib.pyplot as plt
# plt.figure(figsize=(9, 6))
# ax = sns.heatmap(cm, annot=True, cmap='Blues')

# ax.set_title('Gussian outlier detector Confusion Matrix Accuracy = ' + str(round(accuracy, 2)) + ' %' + '\n\n')
# ax.set_xlabel('\nPredicted Values')
# ax.set_ylabel('Actual Values ')
#ax.xaxis.set_ticklabels(['africa','conv','fact']); ax.yaxis.set_ticklabels(['africa', 'conv','fact']);

# annotationsTest=["A1","A1","A1","A1","A1","A1","A2","A2","A2","A2","A2","A2","C1","C1","C1","C1","C1",
#                 "C1","C2","C2","C2","C2","C2","C2","C3","C3","C3","C3","C3","C3","C8","C8","C8","C8",
#                 "C8","C8","F1","F1","F1","F1","F1","F1"]

# fig, ax = plt.subplots(figsize=(12,6))
# ax.scatter(AdagioTrain['feature1'], AdagioTrain['feature2'],label='Training set')
# ax.scatter(AdagioTest['feature1'], AdagioTest['feature2'],label='Test set')
# plt.legend()
# ax.set_xlabel('0-1kHz')
# ax.set_ylabel('1.5kHz')
# for i, label in enumerate(annotationsTest):
#     plt.annotate(label, (AdagioTest['feature1'][i], AdagioTest['feature2'][i]))







from scipy.stats import norm
# Plot between -10 and 10 with .001 steps.
#x_axis = np.arange(-1, 4, 0.01)
# Calculating mean and standard deviation
# meanTest = AdagioTest['feature1'].mean()
# sdTest = AdagioTest['feature1'].std()
# plt.plot(x_axis, norm.pdf(x_axis, meanTest, sdTest))

# meanTrain = AdagioTrain['feature1'].mean()
# sdTrain = AdagioTrain['feature1'].std()
# plt.plot(x_axis, norm.pdf(x_axis, meanTrain, sdTrain))




