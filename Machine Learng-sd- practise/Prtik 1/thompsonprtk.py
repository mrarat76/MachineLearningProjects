# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 18:44:32 2023

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
birler = [0]*d
sifirlar = [0]*d
for n in range (0,N):
   ad = 0 ##seçilen ilan
   max_th=0
   for i in range(0,d):
       rasbeta = random.betavariate (birler[i]+1, sifirlar [i] +1 )
       if rasbeta > max_th:
           max_th = rasbeta
           ad=i
   secilenler.append(ad)     
   odul = veriler.values[n,ad]
   if odul ==1:
       birler[ad] = birler[ad]+1
   else:
       sifirlar[ad]= sifirlar[ad]+1
   oduller[ad]= oduller[ad]+odul
   toplam = toplam + odul    
    
   toplam = toplam+odul
      
''' 
      if(tiklamalar[i] > 0):
             ortalama = oduller[i] / tiklamalar[i]
             delta = math.sqrt(3/2*math.log(n)/tiklamalar[i]) ##ucb formülizasyonu
             ucb = ortalama + delta
      else:
             ucb= N*10
      if max_ucb < ucb: ##maxtan büyük ucb çıktı
            max_ucb=ucb
            ad=i
'''    
            

        
print('Toplam Odul:')

print(toplam)    


plt.hist(secilenler) 
plt.show()       