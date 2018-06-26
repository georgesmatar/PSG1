#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 11:15:09 2017

@author: tarek
"""

import pandas as pd 
import numpy as np 


data_base_path='/home/tarek/Projet_sleep_age/DB TRIO MONTREAL.xlsx'
df=pd.read_excel(data_base_path)

select_data='Data_all'
condition='age_gender'
cutoff_age=41


if select_data=='Data_all':
    
    nb_total=df.shape[0]
    
    df_MCI=df.loc[df['MCI']==1]
    nb_MCI=df.loc[df['MCI']==1].shape[0]
    
    df_AHI=df.loc[df['AHI_Group']==1]
    nb_AHI=df.loc[df['AHI_Group']==1].shape[0]
    
    df_TBI=df.loc[df['TBI']==1]
    nb_TBI=df.loc[df['TBI']==1].shape[0]
   
    df_SOMANA=df.loc[df['Project']=='SOMANA']
    nb_SOMANA=df.loc[df['Project']=='SOMANA'].shape[0]

    nb_man=df.loc[df['Gender']==1].shape[0]
    nb_women=df.loc[df['Gender']==2].shape[0]

    df_young_m_all=df.loc[(df['AgeAtPSGrecording']<cutoff_age) & (df['Gender']==1)]
    nb_young_m_all=df_young_m_all.shape[0]
    df_old_m_all=df.loc[(df['AgeAtPSGrecording']>=cutoff_age) & (df['Gender']==1)]
    nb_old_m_all=df_old_m_all.shape[0]                    
    df_young_w_all=df.loc[(df['AgeAtPSGrecording']<cutoff_age) & (df['Gender']==2)]
    nb_young_w_all=df_young_w_all.shape[0]
    df_old_w_all=df.loc[(df['AgeAtPSGrecording']>=cutoff_age) & (df['Gender']==2)]
    nb_old_w_all=df_old_w_all.shape[0]

    print ('TRIO data base all subject Projects informations : ')
    print('Nombre total des sujets TRIO: ', nb_total)
    print('Nombre des sujets TRIO_TBI: ', nb_TBI)
    print('Nombre des sujets TRIO_MCI: ', nb_MCI)
    print('Nombre des sujets TRIO_AHI: ', nb_AHI)
    print('Nombre des sujets SOMANA: ', nb_MCI)


    print ('TRIO data base all subject age and gender informations : ')
    print ('Nombre total des hommes: ', nb_man)
    print ('Nombre total des jeunes hommes: ', nb_young_m_all)
    print ('Nombre total des hommes ages : ', nb_old_m_all)
    print ('Nombre total des femmes: ', nb_women)
    print ('Nombre total des jeunes femmes: ', nb_young_w_all)
    print ('Nombre total des femmes agees : ', nb_old_w_all)

elif select_data=='Control_all':
    Ctrl_df=df.loc[(df['MCI']!=1) & (df['AHI_Group']!=1 ) & (df['TBI']!=1) ] 
    if condition=='age_gender': 
        
        Gend_nan_indx=np.where(Ctrl_df['Gender'].isnull()==True) 
        Ctrl_df=Ctrl_df.drop(Ctrl_df.index[Gend_nan_indx])
# all data 
        df_man=Ctrl_df.loc[Ctrl_df['Gender']==1]
        nb_woman_all=Ctrl_df.loc[Ctrl_df['Gender']==2].shape[0]
        nb_man_all=Ctrl_df.loc[Ctrl_df['Gender']==1].shape[0]
        df_young_m_all=Ctrl_df.loc[(Ctrl_df['AgeAtPSGrecording']<cutoff_age) & (Ctrl_df['Gender']==1)]
        nb_young_m_all=df_young_m_all.shape[0]
        df_old_m_all=Ctrl_df.loc[(Ctrl_df['AgeAtPSGrecording']>=cutoff_age) & (Ctrl_df['Gender']==1)]
        nb_old_m_all=df_old_m_all.shape[0]                    
        df_young_w_all=Ctrl_df.loc[(Ctrl_df['AgeAtPSGrecording']<cutoff_age) & (Ctrl_df['Gender']==2)]
        nb_young_w_all=df_young_w_all.shape[0]
        df_old_w_all=Ctrl_df.loc[(Ctrl_df['AgeAtPSGrecording']>=cutoff_age) & (Ctrl_df['Gender']==2)]
        nb_old_w_all=df_old_w_all.shape[0]
##data with subjective measures                
        nan_indx=np.where(Ctrl_df['PSQITotalScore'].isnull()==True) 
        database=Ctrl_df.drop(Ctrl_df.index[nan_indx])
        nb_woman=database.loc[database['Gender']==2].shape[0]
        nb_man=database.loc[database['Gender']==1].shape[0]
        df_young_m=database.loc[(database['AgeAtPSGrecording']<cutoff_age) & (database['Gender']==1)]
        nb_young_m=df_young_m.shape[0]
        df_old_m=database.loc[(database['AgeAtPSGrecording']>cutoff_age) & (database['Gender']==1)]
        nb_old_m=df_old_m.shape[0]          
        df_young_w=database.loc[(database['AgeAtPSGrecording']<cutoff_age) & (database['Gender']==2)]
        nb_young_w=df_young_w.shape[0]
        df_old_w=database.loc[(database['AgeAtPSGrecording']>cutoff_age) & (database['Gender']==2)]
        nb_old_w=df_old_w.shape[0]
               
        print ('with subjectives data : \n')
        print ('Nombre total des hommes: ', nb_man)
        print ('Nombre total des jeunes hommes: ', nb_young_m)
        print ('Nombre total des hommes agés : ', nb_old_m)
        print ('Nombre total des femmes: ', nb_woman)
        print ('Nombre total des jeunes femmes: ', nb_young_w)
        print ('Nombre total des femmes agées : ', nb_old_w)
    
