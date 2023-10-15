# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 22:32:37 2023

@author: Monster
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



veriler = pd.read_csv('C:/Users/Monster/Desktop/Şadi Makine/Prtik 1/bilkav.com_maaslar.csv')


x = veriler[['Egitim Seviyesi']]
y= veriler[['maas']]

X = x.values
Y= y.values

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_olcekli = sc1.fit_transform(X)
sc2 = StandardScaler()
y_olcekli = sc2.fit_transform(Y)



from sklearn.svm import SVR

svr_reg = SVR(kernel = 'rbf')

svr_reg.fit(x_olcekli, y_olcekli)

plt.scatter(x_olcekli, y_olcekli)


plt.plot(x_olcekli, svr_reg.predict(x_olcekli), color ='blue' )