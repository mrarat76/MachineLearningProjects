# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 22:01:04 2023

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







## Decision Tree

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)

r_dt.fit(X, Y)

plt.scatter(X,Y, color='green')

plt.plot(X,r_dt.predict(X),color = 'red')

plt.show()


## Random Forest


from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(n_estimators =10 , random_state=0) ## estimator kaç adet decision tree yapacağını karar vermek için burada 10 dedik


rf_reg.fit(X,Y.ravel())

Z = X+0.5
K= X-0.4
plt.scatter(X, Y, color='green')
plt.plot(X,rf_reg.predict(X), color = 'purple')

plt.plot(X,rf_reg.predict(Z), color= 'red')


## R^2 değeri
from sklearn.metrics import r2_score

print('Random Forest R2 değeri')
r2_score(Y, rf_reg.predict(X))

## ynin gerçek değeri ve tahmin yazılır.