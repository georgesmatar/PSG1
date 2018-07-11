#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 14:01:44 2018

@author: tarek
"""

import pandas as pd 


#import collections

file1='/media/tarek/CEAMS_database/EEG_Classification/RECAP_edf_all.xlsx'

df1=pd.read_excel(file1)
subj=df1['Subject_id'].tolist()
list_all=df1['Subject_id'].tolist()
seen = set()
doublons,idx_rem=[],[]
for idx, x in enumerate(subj):
#for x in subj:
    #print 'x', x
    list_all.remove(x)
    for j in list_all:
        
        if x in j:
            print 'x',x,'j',j
            #df1.drop([idx])x
            idx_rem.append(idx)
#        doublons.append(x)
#        seen.add(x)

    
df2=df1.drop(idx_rem)


#print [item for item, count in collections.Counter().items() if count > 1]
#       
#       
#seen = set()
#uniq = []
#for x in subj:
#    if x not in seen:
#        uniq.append(x)
#        seen.add(x)