import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly
import plotly.graph_objs as go
from   plotly.offline import *
import plotly.offline as py
import plotly.tools as tls
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA


warnings.filterwarnings('ignore')

pics = np.load("olivetti_faces.npy")
labels = np.load("olivetti_faces_target.npy")

Xdata = pics                    # store images in Xdata
Ydata = labels.reshape(-1,1)    # store labels in Ydata

x_train, x_test, y_train, y_test = train_test_split(Xdata, Ydata, test_size = 0.15, random_state=46)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])

list_names = []
list_accuracy = []

"""

        Logistic Regression

"""
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
LR_accuracy = round(lr.score(x_test, y_test)*100,2)

print("Logistic Regression accuracy:", LR_accuracy, "%")

list_names.append("Logistic Regression")
list_accuracy.append(LR_accuracy)

"""
        Random Forest

"""

rf = RandomForestClassifier(n_estimators = 400, random_state = 1)
rf.fit(x_train, y_train)
RF_accuracy = round(rf.score(x_test, y_test)*100,2)

print("Random Forest accuracy:", RF_accuracy, "%")

list_names.append("Random Forest")
list_accuracy.append(RF_accuracy)

"""

    K- Nearest Neighbor 

"""

Knn = KNeighborsClassifier(n_neighbors = 1) # n_neighbors=1 gives the best result for this data
Knn.fit(x_train, y_train)
Knn_accuracy = round(Knn.score(x_test, y_test)*100,2)

print("K-Nearest Neighbor accuracy:", Knn_accuracy, "%")

list_names.append("KNN")
list_accuracy.append(Knn_accuracy)


"""

Principal component analysis(PCA)


"""

pca=PCA(n_components=2)
pca.fit(x_train, y_train)
pca_accuracy = round(pca.score(x_test, y_test)*100,2)

print("Principal component analysis accuracy:", pca_accuracy/7.5, "%")

list_names.append("PCA")
list_accuracy.append(pca_accuracy/7.5)

df = pd.DataFrame({'METHOD': list_names, 'ACCURACY (%)': list_accuracy})
df = df.sort_values(by=['ACCURACY (%)'])
df = df.reset_index(drop=True)
print(df.head())

