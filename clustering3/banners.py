# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:57:17 2018

@author: Inspiron
"""

import os
import pandas as ps
import numpy as np
import matplotlib.pyplot as plt



#os.chdir("c:\\Luna\\Work\\python\\coursera\\clustering3\\")
#%%

df  = ps.read_table("checkins.dat", sep = "|", skipinitialspace = True, skiprows = 2,  \
                    names=['id','user_id','venue_id', 'latitude', 'longitude', 'created_at'], skip_blank_lines = False )
                    #usecols = ['id    ', 'user_id ', 'venue_id ', 'latitude      ', 'longitude     ',       'created_at      '])

 
                    #usecols =['id','user_id','venue_id', 'latitude', 'longitude', 'created_at'])
#df =  df[(df.latitude != 'NaN')]