# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 16:15:38 2016

@author: Luna
"""

import numpy as np
import re 

f = open("sentences.txt","r")

sentences = []
words = []


i = 0
for line in f:    
    sentences.append(line.lower())
    words.append( re.split('[^a-z]', sentences[i]))
    for word in words[i]:
        if word ==0
    i = i + 1
f.close()
