import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#%%
#data = pd.read_csv("ghcnd-stations.csv")
#41876           MATVEEV KURGAN
#41882    TAGANROG (LIGHTHOUSE)
#45734             KRASNY LIMAN
#45742                ARTEMIVSK
#45744                  DONETSK
#45746               DEBALTSEVE
#45752               VOLNOVAKHA
#45753       VELIKO-ANADOL'SKOE
#45754              AMVROSIIVKA
#45757                MARIUPOL'
#Name: NAME, dtype: object
#ID           UPM00034519
#LATITUDE          48.067
#LONGITUDE         37.767
#ELEVATION            225
#STATE                NaN
#NAME             DONETSK
#GSNFLAG              NaN
#HCNFLAG              NaN
#WMOID              34519
#Name: 45744, dtype: object
#n1 = np.logical_and(data.LATITUDE>47, data.LATITUDE <49)
#n2 = np.logical_and(data.LONGITUDE>37, data.LONGOTUDE <39)

#adcfac174f7b2cf70abf891467dd2bc08d12663cbb95d5

#%%
data = pd.read_csv("adcfac174f7b2cf70abf891467dd2bc08d12663cbb95d5cd4e348b96.csv")
#%%
data["Date"] = pd.to_datetime(data["Date"])

def chdate(line):
    #print(line)
   #date= line["Date"].to_datetime()
    date=line["Date"]
    ret = [date.year,date.month*100+date.day]
    line["y"]  = ret[0]
    line["d"] = ret[1]
    return line
data = data.apply(chdate,axis = 1)
data = data.drop(data[data.d == 229].index)
#%%
tmin = pd.DataFrame()
tmax = pd.DataFrame()

for y in np.arange(2005, 2015):
    CritMIN = np.logical_and(data.Element == "TMIN", data.y == y)
    CritMAX = np.logical_and(data.Element == "TMAX", data.y == y)    
    tmin.append(data.loc[:,"Data_Value"][CritMIN])
    tmax.append(data.loc[:,"Data_Value"][CritMAX])
    
#df = pd.DataFrame({tmin
    
plt.plot(a)