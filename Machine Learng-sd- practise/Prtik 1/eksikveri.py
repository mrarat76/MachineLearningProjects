# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#kutuphaneler


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## VERİ ÖNİŞLEME
#kodlar
#veri yukleme

veriler= pd.read_csv("eksikveriler.csv")

print(veriler)


#veri on isleme
boy= veriler [["boy"]]
print(boy)

boykilo = veriler [["boy", "kilo"]]
print(boykilo)

#eksikveriler
from sklearn.impute import SimpleImputer

imputer =SimpleImputer(missing_values= np.nan, strategy= "mean")


Yas = veriler.iloc[:, 1:4].values 

print(Yas)

imputer=imputer.fit(Yas[:,1:4])  ## verileri imputer metodu ile ekler.

Yas[:, 1:4] = imputer.transform(Yas[:, 1:4]) ## eklenmiş verilerin bulunduğu imputerla orijinali değiştirir ve salt olarak eklenmiş olur.
print(Yas)
## encoder  : Kategorik -> Numerik
ulke= veriler.iloc[:, 0:1].values

print(ulke)



from sklearn import preprocessing

le= preprocessing.LabelEncoder() ##0,1,2 diye kodlar sütundaki verileri.

ulke [:,0] = le.fit_transform(veriler.iloc[:,0])

print(ulke)

ohe= preprocessing.OneHotEncoder() ## [0,0,1] diye 1,2 diye kodlanan verileri 0 ve 1 kodlarına çevirir.
ulke = ohe.fit_transform(ulke).toarray()

print(ulke)

print(list(range(22)))

sonuc =pd.DataFrame(data=ulke, index= range(22), columns= ['fr','tr','us'])

print(sonuc)

sonuc2= pd.DataFrame(data=Yas, index= range(22), columns=['boy','kilo','yas'])
print(sonuc2)


cinsiyet=veriler.iloc[:,-1].values
print(cinsiyet)

sonuc3 = pd.DataFrame(data=cinsiyet, index = range(22), columns=['cinsiyet'])
print(sonuc3)


s= pd.concat([sonuc,sonuc2], axis=1)
print(s)



s2=pd.concat([s, sonuc3], axis=1)
print(s2)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(s,sonuc3, test_size= 0.33, random_state=0 )



from sklearn.preprocessing import StandardScaler


sc=StandardScaler()  ## farklı verileri ölçeklendirerek birbirine benzetir.

X_train= sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)



