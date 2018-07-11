#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:21:04 2018

@author: tarek
"""

#Recap EDF files

#import numpy as np
#import scipy.io as sio
#from mne.io import read_raw_edf
import os
import pandas as pd 
from SleepCodes import extract_data_fromEDF,select_data
#set path to your EDF files

#path='/media/tarek/CEAMS_database/EEG_Classification/Jessica_db/Women/Middle_aged/EDF/' 

path='/media/tarek/CEAMS_database/EEG_Classification/TRIO_db/Controls/EDF/SOMANA/'
subject_names,montage,channel_names,Fs,nb_ch=[],[],[],[],[]


coln=('Subject_id','Number of EEG channels','EEG channel names',
      'Sampling frequency','Montage rejet artefact')
df=pd.DataFrame(columns=coln)

for root, dirs, files in os.walk(path):
    for filename in files:
        if filename.endswith(".edf"):
            i=filename.find('Stade') 
            j=filename.find('Stage') 
            
            if i>0: 
                idx=i 
                
            else: 
                idx=j
               
            
            subject_names.append(unicode(filename[:idx-1],'utf-8'))
            montage.append(unicode(filename[idx+6:-4],'utf-8'))
            data,ch,fs=extract_data_fromEDF((root+filename))
            eeg,selected_channel_names=select_data(data,channels_names=ch,data_to_select='EEG')
            del data, eeg 
            channel_names.append(selected_channel_names)
            nb_ch.append(len(selected_channel_names))
            Fs.append(fs)
            #edfpath.append(unicode(root+filename,'utf-8'))

            
df['Subject_id']=subject_names
df['Number of EEG channels']=nb_ch
df['EEG channel names']=channel_names
df['Sampling frequency']=Fs
df['Montage rejet artefact']=montage
#df['Full path']=edfpath
df.to_excel(path+'EDF_recap.xlsx')