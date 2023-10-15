# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 22:47:14 2023

@author: Monster
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


veriler = pd.read_csv('C:/Users/Monster/Desktop/Şadi Makine/Prtik 1/bilkav.com_maaslar.csv')


x = veriler[['Egitim Seviyesi']]
y= veriler[['maas']]
## Altta x.values metodu kullanacağız o numpy arraye dönüştürmek için dataframedan.
## polinom regresyonudur ama normal lineer regresyonda ne yapacağını görmek istiyoruz.

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(x.values,y.values)


plt.scatter(x, y, lin_reg.predict(x))
plt.scatter(x.values, y.values, color='red')
plt.plot(x,lin_reg.predict(x.values), color ='green')
plt.show()
##polynomial regressiom

## 2. dereceden
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree =2)

x_poly = poly_reg.fit_transform(x.values)

print(x_poly)

lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly,y)

plt.scatter(x.values,y.values, color= 'red')
plt.plot(x.values,lin_reg2.predict(poly_reg.fit_transform(x.values)),color = 'blue')

##4. dereceden
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree =4)

x_poly = poly_reg.fit_transform(x.values)

print(x_poly)

lin_reg2 = LinearRegression()

lin_reg2.fit(x_poly,y)

plt.scatter(x.values,y.values, color= 'red')
plt.plot(x.values,lin_reg2.predict(poly_reg.fit_transform(x.values)),color = 'blue')

