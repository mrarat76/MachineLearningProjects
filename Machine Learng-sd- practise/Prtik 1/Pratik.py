# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 06:33:05 2023

@author: Monster
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


veriler= pd.read_csv('C:/Users/Monster/Desktop/Şadi Makine/Prtik 1/bilkav.com_odev_tenis.csv')

print(veriler)

hdurumu= veriler.iloc[:, 1].values
print(hdurumu)
play = veriler.iloc[:,-1].values
print(play)

windy = veriler.iloc[:,-2].values
print(windy)

## veri ön işleme

from sklearn import preprocessing
le = preprocessing.LabelEncoder()

# Sıcaklık ve nem sütunlarını aynı LabelEncoder ile işleme
veriler['outlook'] = le.fit_transform(veriler['outlook'])
print(veriler['outlook'])

veriler['windy'] = le.fit_transform(veriler['windy'])
print(veriler['windy'])

veriler['play'] = le.fit_transform(veriler['play'])
print(veriler['play'])



from sklearn.preprocessing import OneHotEncoder

# Initialize the OneHotEncoder
ohe = OneHotEncoder(sparse=False)

# Fit and transform the 'outlook' column
outlook_encoded = ohe.fit_transform(veriler[['outlook']])

# Create feature names for the one-hot encoded columns
outlook_categories = ohe.categories_[0]
feature_names = [f'outlook_{category}' for category in outlook_categories]

# Create a DataFrame from the one-hot encoded values with appropriate column names
outlook_encoded_df = pd.DataFrame(outlook_encoded, columns=feature_names)

# Concatenate the one-hot encoded DataFrame with the original DataFrame
veriler = pd.concat([veriler, outlook_encoded_df], axis=1)

# Drop the original 'outlook' column
veriler.drop('outlook', axis=1, inplace=True)

# Print the updated DataFrame
print(veriler)

veriler.rename(columns={'outlook_2': 'sunny', 'outlook_0': 'overcast','outlook_1':'rainy'}, inplace=True)

testkumesi= veriler.iloc[:,1:7]
trainkumesi= veriler.iloc[:,0]


from sklearn.model_selection import train_test_split

x_train, x_test, y_train,y_test = train_test_split(testkumesi,trainkumesi, test_size= 0.33, random_state=0 )


from sklearn.linear_model import LinearRegression

regressor =LinearRegression()

regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

##veriler2 = veriler.apply(preproccessşng.LabelEncoder().fit_transform) kullanarak bütün kolonları label encode yapabilirsin.
veriler2 = veriler.drop(columns=['play'])
##backward elimination.

import statsmodels.api as sm
'''
X= np.append(arr = np.ones((14,1)).astype(int), values= veriler.iloc[:,1:-1], axis=1)

X_1 = veriler2.iloc[:,[1,2,3,4,5,6]].values
X_1= np.array(X_1, dtype=float)
model= sm.OLS(veriler['temperature'],X_1).fit()
print(model.summary())
'''
X= np.append(arr = np.ones((14,1)).astype(int), values= veriler.iloc[:,1:-1], axis=1)

X_1 = veriler2.iloc[:,[1,2,3,4,5]].values
X_1= np.array(X_1, dtype=float)
model= sm.OLS(veriler2['temperature'],X_1).fit()
print(model.summary())




'''
##one hot encoder
ohe= preprocessing.OneHotEncoder(sparse=False)
# Fit and transform the 'outlook' column
outlook_encoded = ohe.fit_transform(veriler[['outlook']])

# Create a DataFrame from the one-hot encoded values
outlook_encoded_df = pd.DataFrame(outlook_encoded, columns=['outlook'])

# Drop the original 'outlook' column
veriler.drop('outlook', axis=1, inplace=True)

# Concatenate the one-hot encoded DataFrame with the original DataFrame
veriler = pd.concat([veriler, outlook_encoded_df], axis=1)



# Print the updated DataFrame
print(veriler)
'''



'''
veriler['temperature'] = le.fit_transform(veriler['temperature'])
veriler['humidity'] = le.fit_transform(veriler['humidity'])

print(veriler)

# Sıcaklık ve nem sütunlarını birleştirme
sicaknem = veriler[['temperature', 'humidity']]
print(sicaknem)

# One-Hot Encoding işlemi
ohe= preprocessing.OneHotEncoder()
sicaknem_encoded = ohe.fit_transform(sicaknem).toarray()
print(sicaknem_encoded)
'''

##backward ile eğitim

testkumesi2= veriler2.drop(columns= ['temperature'])
trainkumesi2= veriler2['temperature']


from sklearn.model_selection import train_test_split

x_2_train, x_2_test, y_2_train,y_2_test = train_test_split(testkumesi2,trainkumesi2, test_size= 0.33, random_state=0 )


from sklearn.linear_model import LinearRegression

regressor =LinearRegression()

regressor.fit(x_2_train, y_2_train)

y_2_pred = regressor.predict(x_2_test)












