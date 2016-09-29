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
    j = 0
    pop_elements = []
    for word in words[i]:
        if len(word) <= 0: # can by  0 1
            pop_elements.append(j)
        j+=1
        
    pop_elements.reverse()
    for x in pop_elements:
        words[i].pop(x)
    i = i + 1

dictionary = {}
num = 0
for sentence in words:       
   for word in sentence:                    
      if word not in dictionary:          
         dictionary.setdefault(word,num)
         num += 1


D = np.zeros((len(sentences), len(dictionary)))

for i in range(len(words)):
    for word in words[i]:
        D[i][dictionary[word]] += 1
        
R = np.zeros(len(sentences))
from scipy import spatial

for i in range(1,len(sentences)):
    R[i] = spatial.distance.cosine(D[0],D[i])

first = np.argmin(R[1:])+1
temp = R[first]
R[first] = 1
second = np.argmin(R[1:])+1
R[first] = temp
f.close()

f = open("submission-1.txt", "w")
f.write(str(first))
f.write(" ")
f.write(str(second))
f.close()
