# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 17:35:07 2023

@author: Monster
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler= pd.read_csv('C:/Users/Monster/Desktop/Şadi Makine/Prtik 1/ads CTR optimisation.csv')

import random

N= 10000
d=10

#Ri(n)
oduller = [0]*d
toplam = 0 #toplam odul
#Ni(n)
tiklamalar = [0] * d 


toplam=0
secilenler = []
'''
for n in range(0,N):
    ad= random.randrange(d)
    secilenler.append(ad)
    odul = veriler.values[n,ad] # verilerdeki n.satır= 1 ise odul 1
    toplam= toplam + odul
    
    
    
plt.hist(secilenler)

plt.show()    
'''
for n in range (0,N):
   ad = 0 ##seçilen ilan
   max_ucb=0
   for i in range(0,d):
      if(tiklamalar[i] > 0):
             ortalama = oduller[i] / tiklamalar[i]
             delta = math.sqrt(3/2*math.log(n)/tiklamalar[i]) ##ucb formülizasyonu
             ucb = ortalama + delta
      else:
             ucb= N*10
      if max_ucb < ucb: ##maxtan büyük ucb çıktı
            max_ucb=ucb
            ad=i
            
            
   secilenler.append(ad)    
   odul = veriler.values[n,ad]
   toplam = toplam+odul
        
print('Toplam Odul:')

print(toplam)    


plt.hist(secilenler) 
plt.show()       