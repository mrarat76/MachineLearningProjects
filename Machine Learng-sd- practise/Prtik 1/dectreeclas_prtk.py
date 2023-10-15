# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 20:17:31 2023

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


from sklearn.tree import DecisionTreeClassifier

dtc= DecisionTreeClassifier(criterion = 'entropy')


dtc.fit(X_train,y_train)

y_pred = dtc.predict(X_test)


## confusion matrisi
from sklearn.metrics import confusion_matrix
print('DTC')
cm= confusion_matrix(y_test, y_pred)

print(cm)

from sklearn.ensemble import RandomForestClassifier

rfc= RandomForestClassifier(n_estimators= 10, criterion='entropy')

rfc.fit(X_train, y_train)

y_predrfc= rfc.predict(X_test)
print('RFC')
cm=confusion_matrix(y_test, y_pred)
print(cm)




