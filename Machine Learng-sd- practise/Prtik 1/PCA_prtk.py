# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 21:53:50 2023

@author: Monster
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler= pd.read_csv('C:/Users/Monster/Desktop/Şadi Makine/Prtik 1/bilkav.com_Wine.csv')

X= veriler.iloc[:, 0:13].values
y= veriler.iloc[:,13].values



#eğitim ve test kümelerinin bölünmesi


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


from sklearn.preprocessing import StandardScaler

sc= StandardScaler()

X_train=sc.fit_transform(X_train)
X_test =sc.transform(X_test)


#PCA

from sklearn.decomposition import PCA

pca= PCA(n_components=2)

X_train2= pca.fit_transform(X_train)
X_test2= pca.transform(X_test)

from sklearn.linear_model import LogisticRegression



#Pca dön. önce
clasifier= LogisticRegression(random_state=0)
clasifier.fit(X_train,y_train)

#PCA dönüşümünden sonra

clasifier2 = LogisticRegression(random_state=0)
clasifier2.fit(X_train2,y_train)

#predicts

y_pred=clasifier.predict(X_test)

y_pred2=clasifier2.predict(X_test2)


from sklearn.metrics import confusion_matrix
print("gerçek pcasız")
cm= confusion_matrix(y_test,y_pred)
print(cm)

#gerçek pca ile

print("gerçek pca ile")
cm2= confusion_matrix(y_test,y_pred2)
print(cm2)

print("pcasız ve pcalı")

cm3= confusion_matrix(y_pred, y_pred2)
print(cm3)



##LDA


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda=LDA(n_components=2)

X_train_lda= lda.fit_transform(X_train, y_train) ###İkisini veriyooruz ki sınıfları ayrıştırsın. Maksimizie etsin.

X_test_lda=lda.transform(X_test)

#LDA donusumunden sonra

clasfiwer_lda=LogisticRegression(random_state=0)

clasfiwer_lda.fit(X_train_lda,y_train)

#Lda verisi tahmini

y_pred_lda= clasfiwer_lda.predict(X_test_lda)

## Lda sonrası/ orijinal veri karşılaştıması
print("Lda ve orijinal")

cm4= confusion_matrix(y_pred, y_pred_lda)

print(cm4)


















