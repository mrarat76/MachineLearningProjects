# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 19:21:33 2023

@author: Monster
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('C:/Users/Monster/Desktop/Şadi Makine/Prtik 1/bilkav.com_musteriler.csv')


X= veriler[['Hacim','Maas']]


from sklearn.cluster import KMeans

kmeans = KMeans (n_clusters=3, init = 'k-means++')
kmeans.fit(X)


print(kmeans.cluster_centers_)



sonuclae =[]
for i in range(1,10):
    kmeans = KMeans (n_clusters=i, init='k-means++', random_state =123)
    kmeans.fit(X)
    sonuclae.append(kmeans.inertia_)
    
    
    

plt.plot(range(1,10),sonuclae)
    
    