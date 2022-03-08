# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 12:52:41 3636

@author: yaara.shapira
"""

import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.stats as stat
from mne.viz import circular_layout, plot_connectivity_circle
import array as arr
from statsmodels.multivariate.manova import MANOVA
import seaborn as sns


chnames = ['F7-0',

 'C3-0',
 'T7-0',

 'P7-0',
 


 'F8-0',


 'C4-0',
 'T8-0',


 'P8-0',
 
  'F7-1',
 'C3-1',
 'T7-1',
 
 'P7-1',
 
 'F8-1',

 'C4-1',
 'T8-1',

 'P8-1']
subjects=[]
path_to_files =\
    "C:/Users/yaara/Documents/morher-infant behavioral-neural synchrony/Orgenized pipeline for f-2-f/connectivity4to7hz/"
csv_files = os.listdir(path_to_files)
csv_cols = ['channel_1', 'channel_2',"theta", "alpha", "beta", "gamma"]
dfObj = pd.DataFrame(columns=['index','channel_1', 'channel_2',"theta", "alpha", "beta", "gamma"])
meanstheta1=[]
meansalpha1=[]
meansbeta1=[]
meansgamma1=[]
for csv_file in csv_files:
    base_name = os.path.basename(csv_file)
    subject = base_name[0:6]  # for example ZW1234
    condition = base_name[14:16]  # for example BO
    df = pd.read_csv(path_to_files+csv_file, usecols= csv_cols, header=0)
    subjects.append(subject)
    # replace indices with channels names
    df[["channel_1","channel_2"]] =\
        df[["channel_1","channel_2"]].replace(list(range(16)), chnames)
        
    # delete symetric duplicates and reflexive entries
    df=df[df.theta != 0].reset_index(drop=False)
  
    # keep only pairs of channels from different subjects
    df = df[df['channel_1'].str.strip().str[-1] !=\
            df['channel_2'].str.strip().str[-1]].reset_index(drop=True)
    #df=df[(df['channel_1'].str[0] == 'C') | (df['channel_2'].str[0] == 'C')].reset_index(drop=True)

    # meanpersubj1=np.mean(df.theta)
    # meanstheta1.append(meanpersubj1)
    # meanperalpha1=np.mean(df.alpha)
    # meansalpha1.append(meanperalpha1)
    # meanperbeta1=np.mean(df.beta)
    # meansbeta1.append(meanperbeta1)
    # meanpergamma1=np.mean(df.gamma)
    # meansgamma1.append(meanpergamma1)
    
    dfObj = dfObj.append(df, ignore_index = True)
dfObjMother=dfObj.drop(['index'], axis=1)






# for i in range (2,6):
  
F2Fa=pd.DataFrame()


for x in range (0,64):

        F2Fa=pd.concat([F2Fa,dfObjMother[x::64].theta])

F2F2=F2Fa.to_numpy()
F2FM=F2F2.reshape(64,63)
F2FMt=F2FM.transpose()
F2FMtAveraged=F2FMt.mean(axis=0) 

names=['F7-1F7-0',	'F7-1C3-0',	'F7-1T7-0',	'F7-1P7-0',	'F7-1F8-0'	,'F7-1C4-0',	'F7-1T8-0',	'F7-1P8-0',	'C3-1F7-0',	'C3-1C3-0',	'C3-1T7-0',	'C3-1P7-0','C3-1F8-0',	'C3-1C4-0',	'C3-1T8-0',	'C3-1P8-0',	'T7-1F7-0',	'T7-1C3-0',	'T7-1T7-0',	'T7-1P7-0',	'T7-1F8-0',	'T7-1C4-0',	'T7-1T8-0',	'T7-1P8-0',	'P7-1F7-0',	'P7-1C3-0'	,'P7-1T7-0',	'P7-1P7-0',	'P7-1F8-0','P7-1C4-0',	'P7-1T8-0'	,'P7-1P8-0'	,'F8-1F7-0'	,'F8-1C3-0',	'F8-1T7-0',	'F8-1P7-0',	'F8-1F8-0',	'F8-1C4-0',	'F8-1T8-0',	'F8-1P8-0',	'C4-1F7-0',	'C4-1C3-0',	'C4-1T7-0',	'C4-1P7-0',	'C4-1F8-0',	'C4-1C4-0',	'C4-1T8-0',	'C4-1P8-0'	,'T8-1F7-0',	'T8-1C3-0'	,'T8-1T7-0'	,'T8-1P7-0',	'T8-1F8-0',	'T8-1C4-0',	'T8-1T8-0',	'T8-1P8-0',	'P8-1F7-0',	'P8-1C3-0',	'P8-1T7-0'	,'P8-1P7-0',	'P8-1F8-0'	,'P8-1C4-0',	'P8-1T8-0','P8-1P8-0']
plt.bar(names,F2FMtAveraged)
#            
