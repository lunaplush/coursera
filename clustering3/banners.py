# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:57:17 2018

@author: Inspiron
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



#os.chdir("c:\\Luna\\Work\\python\\coursera\\clustering3\\")
#%%

df  = ps.read_table("checkins.dat", sep = "|", skipinitialspace = True, skiprows = 2,  \
                    names=['id','user_id','venue_id', 'latitude', 'longitude', 'created_at'], skip_blank_lines = False )


#df.dropna(subset = ['latitude', 'longitude'], inplace = True)

df = df[~df.latitude.isnull()]


X = np.array(df.loc[:][['latitude','longitude']])[0:100000,:]


#%%
## MEANSHIFT

from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

# #############################################################################
# Generate sample data
#centers = [[1, 1], [-1, -1], [1, -1]]
#X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# #############################################################################
# Compute clustering with MeanShift

# The following bandwidth can be automatically detected using
#bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
bandwidth = 0.1

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

#%%


clustering_res = pd.DataFrame(labels, columns = ["cluster"])

elements_in_cluster = np.zeros(n_clusters_, dtype = "int")
big_clusters = []
j = 0
for i in clustering_res.groupby("cluster"):
   elements_in_cluster[j] =  i[1].shape[0]
   j = j+1   
   if i[1].shape[0] > 15:
       big_clusters.append(i[0])
       
cluster_centers_big_clusters = cluster_centers[big_clusters]   

#%%s
f = open("centers.txt","w")
for i in cluster_centers_big_clusters:
    f.write("{},{} <green>\n".format(i[0],i[1]))
f.close()

#%%
offices = np.array([[33.751277, -118.188740], \
           [25.867736, -80.324116], \
           [51.503016, -0.075479],\
           [52.378894, 4.885084], \
           [39.366487, 117.036146],\
           [-33.868457, 151.205134]])
           
#%%
def dist(x,y):
   return  np.sqrt(np.power(x[0]-y[0],2) + np.power(x[1]-y[1],2))

distance = np.zeros(len(cluster_centers_big_clusters))
for i in np.arange(len(cluster_centers_big_clusters)):
    ds = np.zeros(len(offices))
    for j in np.arange(len(offices)):
        ds[j] = dist(cluster_centers_big_clusters[i],offices[j])
    distance[i] = ds.min() 
    
i = distance.argmin()
#%%
f = open("answer.txt","w")

f.write("{} {}".format(i[0],i[1]))
f.close()

         