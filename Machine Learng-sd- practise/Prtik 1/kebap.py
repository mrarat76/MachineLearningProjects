# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 06:48:26 2023

@author: Monster
"""

import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv(r'C:\Users\Monster\Desktop\Şadi Makine\Prtik 1\satislar.csv',  delimiter=';')
print (veriler)


aylar = veriler [['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(aylar,satislar,test_size = 0.33, random_state=0)
'''
from sklearn.preprocessing import StandardScaler


sc=StandardScaler()  ## farklı verileri ölçeklendirerek birbirine benzetir.

X_train= sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)
Y_train = sc.fit_transform(y_train)
Y_test = sc.fit_transform(y_test)
'''

## model inşaası (linear regression)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()

lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)
x_train = x_train.sort_index()
y_train = y_train.sort_index()
plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.title("satış")
plt.xlabel("aylar")
plt.ylabel("satış")
