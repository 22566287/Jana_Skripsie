import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def save_multi_image(filename):
    pp = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

# import the data
A4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\A4.csv")
D4data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\D4.csv")
E5data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\E5.csv")
G3data = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\G3.csv")
Adagiodata = pd.read_csv("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\inputs\\Adagio.csv")
#print(A4data.head(15))
annotations=["A1","A2","C1","C10","C2","C3","C4","C5","C6","C7","C8","C9","F1","F2","F3"]
targets = ['africa1', 'africa2', 'conv1', 'conv10', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'fact1', 'fact2', 'fact3']
colors = ['r','r','g','g','g','g','g','g','g','g','g','g','b','b','b']

#A
# standardize the data
features = ['feature1', 'feature2', 'feature3', 'feature4']# Separating out the features
xA = A4data.loc[:, features].values# Separating out the target
yA = A4data.loc[:,['violin']].values# Standardizing the features
xA = StandardScaler().fit_transform(xA)

# PCA projection to 2D
pca = PCA(n_components=2)
principalComponentsA = pca.fit_transform(xA)
principalDfA = pd.DataFrame(data = principalComponentsA, columns = ['pc1', 'pc2'])
finalDfA = pd.concat([principalDfA, A4data[['violin']]], axis = 1)

# visualize 2D projection
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('A4 energy PCA ', fontsize = 20)
for target, color in zip(targets,colors):
    indicesToKeep = finalDfA['violin'] == target
    ax.scatter(finalDfA.loc[indicesToKeep, 'pc1']
               , finalDfA.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (finalDfA.loc[i, 'pc1'], finalDfA.loc[i, 'pc2']))
#plt.show()

#D
# standardize the data
features = ['feature1', 'feature2', 'feature3', 'feature4']# Separating out the features
xD = D4data.loc[:, features].values# Separating out the target
yD = D4data.loc[:,['violin']].values# Standardizing the features
xD = StandardScaler().fit_transform(xD)

# PCA projection to 2D
pca = PCA(n_components=2)
principalComponentsD = pca.fit_transform(xD)
principalDfD = pd.DataFrame(data = principalComponentsD, columns = ['pc1', 'pc2'])
finalDfD = pd.concat([principalDfD, D4data[['violin']]], axis = 1)

# visualize 2D projection
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('D4 energy PCA ', fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = finalDfD['violin'] == target
    ax.scatter(finalDfD.loc[indicesToKeep, 'pc1']
               , finalDfD.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (finalDfD.loc[i, 'pc1'], finalDfD.loc[i, 'pc2']))
#plt.show()

#E
# standardize the data
features = ['feature1', 'feature2', 'feature3', 'feature4']# Separating out the features
xE = E5data.loc[:, features].values# Separating out the target
yE = E5data.loc[:,['violin']].values# Standardizing the features
xE = StandardScaler().fit_transform(xE)

# PCA projection to 2D
pca = PCA(n_components=2)
principalComponentsE = pca.fit_transform(xE)
principalDfE = pd.DataFrame(data = principalComponentsE, columns = ['pc1', 'pc2'])
finalDfE = pd.concat([principalDfE, E5data[['violin']]], axis = 1)

# visualize 2D projection
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('E5 energy PCA ', fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = finalDfE['violin'] == target
    ax.scatter(finalDfE.loc[indicesToKeep, 'pc1']
               , finalDfE.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (finalDfE.loc[i, 'pc1'], finalDfE.loc[i, 'pc2']))
#plt.show()

#G
# standardize the data
features = ['feature1', 'feature2', 'feature3', 'feature4']# Separating out the features
xG = G3data.loc[:, features].values# Separating out the target
yG = G3data.loc[:,['violin']].values# Standardizing the features
xG = StandardScaler().fit_transform(xG)

# PCA projection to 2D
pca = PCA(n_components=2)
principalComponentsG = pca.fit_transform(xG)
principalDfG = pd.DataFrame(data = principalComponentsG, columns = ['pc1', 'pc2'])
finalDfG = pd.concat([principalDfG, G3data[['violin']]], axis = 1)

# visualize 2D projection
fig = plt.figure(figsize = (10,7))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('G3 energy PCA ', fontsize = 20)

for target, color in zip(targets,colors):
    indicesToKeep = finalDfG['violin'] == target
    ax.scatter(finalDfG.loc[indicesToKeep, 'pc1']
               , finalDfG.loc[indicesToKeep, 'pc2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
for i, label in enumerate(annotations):
    plt.annotate(label, (finalDfG.loc[i, 'pc1'], finalDfG.loc[i, 'pc2']))
#plt.show()



save_multi_image("C:\\Users\\Jana\\Documents\\Stellenbosch_Ingenieurswese\\Lesings\\2022\\2de_semester\\Project_E_448\\AudioAnalysisofanAfricanViolin\\violinData\\results\\PCAenergyResults.pdf")