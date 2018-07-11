#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 11:51:43 2018

@author: tarek
"""

import pandas as pd 
import numpy as np
import os
path='/media/tarek/CEAMS_database/EEG_Classification/'
projects=['Jessica_db','TRIO_db']
df1=pd.DataFrame([])
db=[]

#print [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
for x in os.walk(path):   
    if x[0].endswith('EDF'):
       
       for p in os.walk(x[0]):
           
           for f in p[2]:
               if f.endswith('.xlsx'):
                   
                   df=pd.read_excel(os.path.join(p[0],f))
                   df1=df1.append(df, ignore_index=True)
              
col=['Subject_id','Number of EEG channels','EEG channel names','Sampling frequency',
'Montage rejet artefact']
df1=df1[col]                 
df1.to_excel(os.path.join(path,'RECAP_edf_all.xlsx'))

               #for p in projects:
    

   