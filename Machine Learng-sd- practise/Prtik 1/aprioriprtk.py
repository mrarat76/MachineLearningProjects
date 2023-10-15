# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 15:54:17 2023

@author: Monster
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import apyori

veriler = pd.read_csv('C:/Users/Monster/Desktop/Şadi Makine/Prtik 1/bilkav.com_sepet.csv', header= None)


t=[]

for i in range (0,7501):
    t.append([str(veriler.values[i,j]) for j in range(0,20)])


from apyori import apriori
kurallar=apriori(t, min_support=0.01,min_confidence= 0.2, min_lift=3, min_lenght=2)



print(list(kurallar))
#apriori(transactions, kwargs)