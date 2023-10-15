# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:20:09 2023

@author: Monster
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler2= pd.read_csv("C:/Users/Monster/Desktop/Şadi Makine/Prtik 1/bilkav.com_veriler.csv")

print(veriler2)

ulke= veriler2.iloc[:, 0:1].values

print(ulke)
Yas = veriler2.iloc[:, 1:4].values 
## encoder  : Kategorik -> Numerik
testcinsiyet= veriler2.iloc[:, -1:].values

print(testcinsiyet)



from sklearn import preprocessing

le= preprocessing.LabelEncoder() ##0,1,2 diye kodlar sütundaki verileri.

testcinsiyet[:,-1] = le.fit_transform(veriler2.iloc[:,-1])
print(testcinsiyet)

ohe= preprocessing.OneHotEncoder() ## [0,0,1] diye 1,2 diye kodlanan verileri 0 ve 1 kodlarına çevirir.
testcinsiyet = ohe.fit_transform(testcinsiyet).toarray()
print(testcinsiyet)
ulke [:,0] = le.fit_transform(veriler2.iloc[:,0])


ulke = ohe.fit_transform(ulke).toarray()

print(ulke)

print(list(range(22)))

sonuc =pd.DataFrame(data=ulke, index= range(22), columns= ['fr','tr','us'])

print(sonuc)

sonuc2= pd.DataFrame(data=Yas, index= range(22), columns=['boy','kilo','yas'])
print(sonuc2)

sonuc3 = pd.DataFrame(data= testcinsiyet[:,:1], index =range(22), columns = ['cinsiyet'])
print(sonuc3)


s= pd.concat([sonuc,sonuc2], axis=1)
print(s)



s2=pd.concat([s, sonuc3], axis=1)
print(s2)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(s,sonuc3, test_size= 0.33, random_state=0 )



from sklearn.linear_model import LinearRegression

regressor =LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)



boy= s2.iloc[:,3:4].values
boy= pd.DataFrame(boy,columns = ['boy'])


print(boy)

sol = s2.iloc[:,:3]
sag = s2.iloc[:,4:]



veri= pd.concat([sol,sag], axis =1)


X_train, X_test, Y_train, Y_test = train_test_split(veri,boy,test_size= 0.33, random_state=0)

r2= LinearRegression()

r2.fit(X_train, Y_train)


y_pred= r2.predict(X_test)

import statsmodels.api as sm

X1 = np.append(arr = np.ones((22,1)).astype(int), values = veri, axis=1)
## çoklu lineer regresyonda sabit değer gerekir a+ax+ax^2 gibi gidiyor onun için sabit değer lazım. 22 tane bir eklliyoruz. Nedeni ise çarpanın 1 olmasıdır.

X_1 = veri.iloc[:,[0,1,2,3,4,5]].values
X_1= np.array(X_1, dtype=float)
model= sm.OLS(boy,X_1).fit()
print(model.summary())
'''
Kodun adımlarını açıklayalım:

İlk olarak, statsmodels kütüphanesinden sm adını kullanarak gerekli modülü içe aktarıyoruz.

X1 adlı bir veri matrisi oluşturuyoruz. Bu matris, bağımsız değişkenleri içerirken, bir sütunu birlerle doldurarak bir sabit terimi (katsayısı) ekliyor. Bu, çoklu doğrusal regresyon analizinde sabit terimi hesaplamak için gereklidir.

X_1 adlı başka bir veri matrisi oluşturuyoruz. Bu matris, bağımsız değişkenleri içerir, ancak sabit terim içermez. Bu matris, regresyon modelini oluşturmak için kullanılacaktır.

model adlı bir OLS (En Küçük Kareler Yöntemi) regresyon modeli oluşturuyoruz. OLS, veriler arasındaki en uygun doğrusal ilişkiyi bulmaya çalışan bir istatistiksel modelleme yöntemidir.

Oluşturulan modeli fit() yöntemiyle verilere uyarız ve uygun regresyon katsayılarını hesaplarız.

model.summary() yöntemi ile modelin özetini yazdırırız. Bu özet, regresyon sonuçlarını, katsayıları, t-testi ve p-değerlerini, R-kare değerini ve diğer önemli istatistikleri içerir. Bu özet, modelin ne kadar iyi uyum sağladığını ve her bağımsız değişkenin bağımlı değişken üzerindeki etkisini değerlendirmemize yardımcı olur.

Bu kod, çoklu doğrusal regresyon analizi sonuçlarını incelemek ve bağımsız değişkenlerin bağımlı değişken üzerindeki etkisini değerlendirmek için kullanılır.
P value bakılır.
'''
