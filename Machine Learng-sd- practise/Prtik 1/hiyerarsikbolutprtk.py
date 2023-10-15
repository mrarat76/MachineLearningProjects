# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 01:21:19 2023

@author: Monster
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('C:/Users/Monster/Desktop/Şadi Makine/Prtik 1/bilkav.com_musteriler.csv')


X= veriler[['Hacim','Maas']]


from sklearn.cluster import AgglomerativeClustering

ac= AgglomerativeClustering(n_clusters=3, affinity= 'euclidean', linkage='ward')


Y_pred= ac.fit_predict(X)
print(Y_pred)

plt.scatter(X.values[Y_pred == 0, 0], X.values[Y_pred == 0, 1], s=100, c='red')
plt.scatter(X.values[Y_pred == 1, 0], X.values[Y_pred == 1, 1], s=100, c='blue')
plt.scatter(X.values[Y_pred == 2, 0], X.values[Y_pred == 2, 1], s=100, c='green')
plt.show()
##plt.scatter(X[Y_pred== 0,0], X[Y_pred==0,1],s=100, c='red')


import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(X,method ='ward'))

plt.show()