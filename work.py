# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 09:55:50 2017

@author: Inspiron
"""
#%%
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.style.use('ggplot')
from scipy import stats

#%%

a = {"name":"A", "state" :"QQQ1"} 
b = {"name":"B", "state" :"QQQ"}
c = {"name":"C", "state" :"QQQ"} 
d = {"name":"D","state" :"QQQ"}
df = pd.DataFrame([a,b,c,d])


#%%
a2 = {"name":"C","state" :"QQQ", "cena": 3} 

b2 = {"name":"D","state" :"QQQ","cena":1}
c2 = {"name":"A", "state" :"QQQ","cena": 2} 
d2 = {"name":"F","state" :"QQQ","cena":50}

df2 = pd.DataFrame([a2,b2,c2,d2]).set_index(["name","state"])

df3 = df.merge(df2, how = "inner", right_index = True, left_on = ["name","state"])
#%%
#df4 = df2.reset_index()
ll = []
i = 0
#df4["non_uni"] = True
df2["non_uni"] = True
for i in range(len(df)):
    Crit = np.logical_and(df2.index[0] == df.iloc[i]["name"], df2.index[1] == df.iloc[i]["state"])
    #Crit = np.logical_and(df4["name"] == df.iloc[i]["name"], df4["state"] == df.iloc[i]["state"] )
    #df4["non_uni"][Crit] = False 
    df2["non_uni"][Crit] = False
    i = i+1
   
df5 = df2[df2["non_uni"]]

#[f,p] = stats.ttest_ind(df,df2)