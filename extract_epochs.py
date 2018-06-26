#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 14:13:57 2018

@author: karim
"""

from SleepCodes import select_data,save_data,extract_data_fromEDF
from SleepCodes import extract_markers,data_epoching,extract_data_per_sleep_stage
import os

path='/home/karim/sleepclassification/'
subject_name='10005n0_caf200_Stage_dépistage-a'
xmlfile=path+subject_name+'.edf.xml'
edf_file=os.path.join(path,subject_name+'.edf')




hyp,Art_ch,Art_start,Art_duration=extract_markers(xmlfile,True,path,subject_name)
data,channel_names,Fs=extract_data_fromEDF(edf_file)
EEG_data,ch_names=select_data(data,channel_names,'EEG')
Segments=data_epoching(data=EEG_data,epoch_length=30,Fs=Fs)
stages=extract_data_per_sleep_stage(Segments,hyp,'AASM')
#save_path=path+subject_name
##save_data(stages,save_path,'npy')
