import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import datetime
import matplotlib.ticker as ticker
import os
from matplotlib.artist import Artist

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
os.chdir("c:\\Luna\\Work\\python\\coursera\\michigan\\")
data = pd.read_csv("adcfac174f7b2cf70abf891467dd2bc08d12663cbb95d5cd4e348b96.csv")
#%%
data["Date"] = pd.to_datetime(data.Date)
data.drop(1957,inplace = True)
def chdate(line):
    #print(line)
   #date= line["Date"].to_datetime()
    date=line["Date"]
   
    if ( date.month == 2 and date.day == 29):
        ku = 1
    else:        
        ret = [date.year,date.month*100+date.day]
        if(date.year !=2015):
            date = date.replace(year = 2005)
        
        line["Date"] = date
        line["y"]  = int(ret[0])
        line["d"] = int(ret[1])  
        line["d2"] = int(ret[1])
        line["Data_Value"] = line["Data_Value"]/10
        return line
data = data.apply(chdate,axis = 1)
data = data.drop(data[data.d == 229].index)
#%%

data2015 = data[data.y.eq(2015)]
data = data[data.y.ne(2015)]
#%%

#tmin = data[data.Element == "TMIN"].groupby("d")["Data_Value"].min()
#tmax = data[data.Element == "TMAX"].groupby("d")["Data_Value"].max()
tmin = data[data.Element == "TMIN"].groupby("d").min()
tmax = data[data.Element == "TMAX"].groupby("d").max()

#%%
#plt.plot(tmin.index, tmin.data)
#plt.plot(tmax.index, tmax.data)
tmin = tmin.sort_values(by = "Date")
tmax = tmax.sort_values(by = "Date")
tmin["flDate"] = tmin.Date.apply(mpl.dates.date2num)
tmax["flDate"] = tmax.Date.apply(mpl.dates.date2num)
#plt.plot(tmin.flDate, tmin.Data_Value, color = "blue")
#plt.plot(tmax.flDate, tmax.Data_Value, color = "red")
#plt.gca().fill_between(tmin.flDate,tmin.Data_Value,tmax.Data_Value,facecolor='m', alpha=0.25)
#%%
data2015["flDate"] = data2015.Date.apply(mpl.dates.date2num)

tmin2015 = data2015[data2015.Element == "TMIN"].groupby("d").min()
tmin2015 = tmin2015.sort_values(by = "Date")

tmax2015 = data2015[data2015.Element == "TMAX"].groupby("d").max()
tmax2015 = tmax2015.sort_values(by = "Date")
v2015 = []
x2015 = []
color = []
for d in tmin2015.Date :
    print(d)
    try:
        vmin2015 = tmin2015[tmin2015.Date == d].iloc[0].Data_Value
        print("vmin2015", vmin2015)
        try:
            print()
            vmin = tmin[tmin.Date == d].iloc[0].Data_Value
        except Exception as inst:
            print(type(inst))     # the exception instance
            print(inst.args)      # arguments stored in .args
            print(inst)         
            print("")
        print("TMIN",vmin2015, vmin)
        if vmin2015 < vmin:
            x2015.append(mpl.dates.date2num(d))
            v2015.append(vmin2015)
            color.append("blue")
    except:
        pass
    try:    
        vmax2015 = tmax2015[tmax2015.Date == d].iloc[0].Data_Value
        vmax = tmax[tmax.Date == d].iloc[0].Data_Value
        print("TMAX",vmax2015,vmax)
        if vmax2015 > vmax:
            x2015.append(mpl.dates.date2num(d))
            v2015.append(vmax2015)
            color.append("red")
    except:
        pass
        
    
#%%
plt.figure(figsize = (10,8))
plt.subplot(111)
points = plt.scatter(x2015,v2015,c = color, marker = "*", label = "days in 2015 that broke a record \n high or low for 2005-2014")    
linemin, = plt.plot(tmin.flDate, tmin.Data_Value, color = "blue", label = "minimum for 2005-2014")
linemax, = plt.plot(tmax.flDate, tmax.Data_Value, color = "red", label = "maximum for 2005-2014")
plt.gca().fill_between(tmin.flDate,tmin.Data_Value,tmax.Data_Value,facecolor='m', alpha=0.25)
plt.title("Record highs and lows temperature  for 2005-2014 in Donetsk \n stations:{},{}".format(data.ID.unique()[0],data.ID.unique()[1]))
plt.ylabel("degrees Celsius")

plt.xlim(tmin.flDate.min(), tmin.flDate.max())
plt.ylim(-40,65)
first_legend = plt.legend(handles = [linemin, linemax,points], loc=1, frameon=False)
                          #bbox_to_anchor=(0.9, 0.9),  bbox_transform=plt.gcf().transFigure)
#second_legend = plt.legend(handles = [points],loc = 3,frameon=False)

x = plt.gca().xaxis



#def rec_gc(art, depth=0):
#    if isinstance(art, Artist):
#        # increase the depth for pretty printing
#        print("  " * depth + str(art))
#        for child in art.get_children():
#            rec_gc(child, depth+2)
#
## Call this function on the legend artist to see what the legend is made up of
#rec_gc(x)




#x.set_major_formatter(ticker.Formatter.format_data(value = "M"))

# rotate the tick labels for the x axis
i = 1
for item in x.get_ticklabels():
    item.set_rotation(45)
    if i in [1,2,12]:
        cl = "blue"
    if i in [3,4,5,9,10,11]:
        cl = "green"
    if i in [6,7,8]:
        cl = "red"       
        
    item.set_color(cl)
    item.set_fontsize(16) 
    i = i + 1
d = []     
for i in np.arange(1,13):
    d.append(mpl.dates.date2num(datetime.datetime(2005,i,15)))
  
m = ['Jan',"Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]  
plt.xticks(d,m )

#plt.text(1,2,"Info")
#plt.show()  
 
#%%

plt.savefig("weather.png")
#tmin = pd.DataFrame()
#tmax = pd.DataFrame()
#for (pare,td) in data.groupby(["y","Element"]):
#    print(len(td))    
#    if  pare[1] =="TMIN":
#        td = td.groupby("d").min()         
#        tmin[y] = td.Data_Value
#        print(td.Data_Value)
#    else:
#        td.groupby("d").max()         
#        tmax[y] =td.Data_Value
    
#%%
#tmin = pd.DataFrame()
#tmax = pd.DataFrame()
#tmin[0] = data[data.Element == "TMIN"].Date.unique()
#tmax[0] = data[data.Element == "TMAX"].Date.unique()
#tmin.set_index(0, inplace = True)
#tmax.set_index(0,inplace = True)
#for y in np.arange(2005, 2015):
#    print(y)
#    CritMIN = np.logical_and(data.Element == "TMIN", data.y == y)
#    CritMAX = np.logical_and(data.Element == "TMAX", data.y == y)   
#    new_a = data.loc[:,["Data_Value","Date"]][CritMIN]
#    new_a.set_index("Date", inplace = True)
#    tmin[y] = new_a
#    
#    new_b = data.loc[:,["Data_Value","Date"]][CritMAX]
#    new_b.set_index("Date", inplace = True)
#    tmax[y]= new_b
    
#df = pd.DataFrame({tmin
    
#plt.plot(a)