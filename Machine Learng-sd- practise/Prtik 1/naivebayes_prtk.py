# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 19:52:29 2023

@author: Monster
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('C:/Users/Monster/Desktop/Şadi Makine/Prtik 1/bilkav.com_veriler.csv')



x= veriler.iloc[:, 1:4].values ##bağımlı değişken
y= veriler.iloc[:,4:].values   ## bağımsız değişken











from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.33, random_state=0)




from sklearn.preprocessing import StandardScaler

sc= StandardScaler()


X_train= sc.fit_transform(x_train) ## fit eğitmek, tranform eğitileni uygulamak.
X_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()


gnb.fit(X_train,y_train)

y_pred= gnb.predict(X_test)

## confusion matrisi
from sklearn.metrics import confusion_matrix

cm= confusion_matrix(y_test, y_pred)

print(cm)