# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 01:28:05 2020

@author: yaara.shapira
"""

import os
import mne
import pandas as pd
import numpy as np
from mne.connectivity import spectral_connectivity as sc
import scipy.stats as stat
import matplotlib.pyplot as plt
# -----------------------------------------------------------------------------
# USER CONFIGURATIONS (SCRIPT CONSTANTS)
#   PATH_DIR_SRC : Path to source files folder
#   PATH_DIR_DEST: Path to target output folder
#   PATH_DURATION_CSV: Path to .csv file containing a duration column, the
#       duration is the minimal time between the different paradigms for each
#       subject. The duration
#   SBJ_COL_NAME : Name of subject names column
#   DUR_COL_NAME : Name of number of epochs column
#   OFFSET_START : Margin of epochs to crop from the beginning of the data
#   OFFSET_END   : Margin of epochs to crop from the end of the data (after
#   TAG          : Paradigm tag
#       cropping the by the duration value)
# -----------------------------------------------------------------------------
PATH_DIR_SRC = "C:/Users/yaara/Documents/morher-infant behavioral-neural synchrony/Free-Interaction/ICAclean/"
PATH_DIR_DEST = "C:/Users/yaara/Documents/morher-infant behavioral-neural synchrony/Free-Interaction/shuffle/"
PATH_DURATION_CSV = "C:/Users/yaara/Documents/morher-infant behavioral-neural synchrony/Free-Interaction/freeint.xlsx"
SBJ_COL_NAME = "file"
DUR_COL_NAME = "EPOCHS NUM"
OFFSET_START =10
OFFSET_END =10
TAG = 'f2f'


#allcon = []
#pall=[]
#allconSloop=[]
Allmeans=[]
# -----------------------------------------------------------------------------
# USER ARGUMENTS VALIDATION
# -----------------------------------------------------------------------------
assert os.path.isdir(PATH_DIR_SRC), "invalid source folder"
assert os.path.isdir(PATH_DIR_DEST), "invalid destination folder"
assert os.path.isfile(PATH_DURATION_CSV), "invalid path to csv file"
assert PATH_DURATION_CSV.endswith(".xlsx"), "invalid file type"
excel_file = pd.read_excel(PATH_DURATION_CSV,engine='openpyxl')
assert SBJ_COL_NAME in excel_file, \
    "csv file does not contain mandatory subject name column"
assert DUR_COL_NAME in excel_file, \
    "csv file does not contain mandatory duration column"
assert isinstance(OFFSET_START, int), "offset should be an integer value"
assert isinstance(OFFSET_END, int), "offset should be an integer value"


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def min_cut_dict() -> dict:
    """
    dict with subject names as keys, and their respective duration as values
    :return: dict object
    """
    excel_file = pd.read_excel(PATH_DURATION_CSV,engine='openpyxl')
    return dict(zip(excel_file[SBJ_COL_NAME], excel_file[DUR_COL_NAME]))


def synch_drop_logs(subj_1, subj_2):
    """
    synchronize drop logs between the two subjects, modified in place
    :param subj_1: mne raw instance
    :param subj_2: mne raw instance
    :return:
    """
    log_1 = subj_1.drop_log
    log_2 = subj_2.drop_log
    clean_1 = [(a, b) for a, b in zip(log_1, log_2) if a == []]
    ind_drop_1 = [i for ((a, b), i) in zip(clean_1, list(range(len(clean_1))))
                  if a == [] and b != []]
    clean_2 = [(a, b) for a, b in zip(log_2, log_1) if a == []]
    ind_drop_2 = [i for ((a, b), i) in zip(clean_2, list(range(len(clean_2))))
                  if a == [] and b != []]
    subj_1.drop(ind_drop_1)
    subj_2.drop(ind_drop_2)


def apply_offset(subj, start, end, duration=None):
    """
    drop 'start' number of epochs from the beginning of the data and 'end' from
    the end
    :param subj: mne raw instacne
    :param start: # of epochs to crop from the start
    :param end:   # of epochs to crop from the end
    :param duration:
    :return:
    """
def min_cut_dict() -> dict:
    """
    dict with subject names as keys, and their respective duration as values
    :return: dict object
    """
    excel_file = pd.read_excel(PATH_DURATION_CSV)
    return dict(zip(excel_file[SBJ_COL_NAME], excel_file[DUR_COL_NAME]))


def synch_drop_logs(subj_1, subj_2):
    """
    synchronize drop logs between the two subjects, modified in place
    :param subj_1: mne raw instance
    :param subj_2: mne raw instance
    :return:
    """
    log_1 = subj_1.drop_log
    log_2 = subj_2.drop_log
    clean_1 = [(a, b) for a, b in zip(log_1, log_2) if not a]
    ind_drop_1 = [i for ((a, b), i) in zip(clean_1, list(range(len(clean_1))))
                  if not a and b]
    clean_2 = [(a, b) for a, b in zip(log_2, log_1) if not a]
    ind_drop_2 = [i for ((a, b), i) in zip(clean_2, list(range(len(clean_2))))
                  if not a and b]
    subj_1.drop(ind_drop_1)
    subj_2.drop(ind_drop_2)


def apply_offset(subj, start, end, duration=None):
    """
    drop 'start' number of epochs from the beginning of the data and 'end' from
    the end
    :param subj: mne raw instacne
    :param start: # of epochs to crop from the start
    :param end:   # of epochs to crop from the end
    :param duration:
    :return:
    """
    log = subj.drop_log
    ind_drop_start = list(range(len([x for x in log[:start] if not x])))
    clean = [x for x in log[:int(duration-end)] if not x]
    clean_total = [x for x in log if not x]
    ind_drop_end = list(range(len(clean), len(clean_total)))
    subj.drop(ind_drop_start+ind_drop_end)


# -----------------------------------------------------------------------------
# MAIN SCRIPT
# ASSUMPTIONS:
#       - For every experiment there are pairs, the naming conventions should
#       be such that sorting the files in A-Z order will yield all pairs
#       adjacent
#       - Original data length is the same for the two members in each pair
# -----------------------------------------------------------------------------
# split files in source folder into pairs (by experiment)
files_ls = os.listdir(PATH_DIR_SRC)
files_ls = [file for file in files_ls if file.endswith('.fif')]
files_ls.sort()
files_ls = np.array(files_ls).reshape(int(len(files_ls)/2), 2)
dic_duration = min_cut_dict()
for esub1 in range(0,8):
    for esub2 in range (8,16):
        averagepercomb=[]
        allconSloop=[]
        for i in range (100):
            allconS = []
# loop over every subject pair
            for (subj_1, subj_2) in files_ls:
    # load and pre-process data
                os.chdir(PATH_DIR_SRC)
                raw_1 = mne.read_epochs(subj_1, preload=True)
                raw_2 = mne.read_epochs(subj_2, preload=True)
                raw_1.pick_types(eeg=True,
 ן                              'Cz', 'Pz', 'Fz'])
                raw_2.pick_types(eeg=True,
                     exclude=['Fp1', 'Fp2', 'O1', 'O2', 'Oz', 'Cz',
                              'Pz', 'Fz'])
                apply_offset(raw_1, OFFSET_START, OFFSET_END, dic_duration[subj_1[0:6]])
                synch_drop_logs(raw_1, raw_2)
                drop_comb = zip(raw_1.drop_log, raw_2.drop_log)

    # preparations for spectral connectivity
                data_1 = raw_1.get_data()
                data_2 = raw_2.get_data()
                data_1 = np.take(data_1,np.random.permutation(data_1.shape[0]),axis=0,out=data_1)
                data_combined = np.concatenate((data_1, data_2), axis=1)
                info = mne.create_info(צתץ,raw_1.info['sfreq'])
                events = np.array([np.array([i, 0, i])
                       for i in range(data_combined.shape[0])])
                events_id = dict(zip([str(x[0]) for x in events], [x[0] for x in events]))
                raw_combined = mne.EpochsArray(data_combined,לצ info, events, -0.5,
                                   events_id)
    
                freq_bands = {'theta': (4, 7)}
                fmin = np.array([f for f, _ in freq_bands.values()])
                fmax = np.array([f for _, f in freq_bands.values()])
    
                indices = (np.array([esub1]),    # row indices
                       np.array([esub2]))    # col indices

    # spectral connectivity
                sc_data = sc(raw_combined, method='wpli', fmin=fmin, fmax=fmax,indices=indices,
                 mode='multitaper', faverage=True, n_jobs=1, verbose=False)
                connB, freqsB, timesB, n_epochsB, n_tapersB = sc_data
                allconS.append(connB.tolist()[0][0])
            allconSloop.append(allconS)
    
   #allcon.append(connB.tolist()[0][0])
       
          
        newfile1=  "Shuffle"+str(esub1)+str(esub2)+".csv"

        newfile1 = os.path.join(PATH_DIR_DEST, newfile1)
        allconSloop=np.transpose(allconSloop)
        pd.DataFrame(allconSloop).to_csv(newfile1)
        averagepercomb=np.mean(allconSloop, axis=1) 
        
        newfile2=  "MeanShuffle"+str(esub1)+str(esub2)+".csv"

        newfile2 = os.path.join(PATH_DIR_DEST, newfile2)
        pd.DataFrame(averagepercomb).to_csv(newfile2)
        averagepercomb=np.transpose(averagepercomb)
        Allmeans.append(averagepercomb)  
#allconSloop.tofile(newfile1,sep=',',format='%10.5f') 
new=np.concatenate( Allmeans, axis=0 )
new2=new.reshape(64,63)
new2=np.transpose(new2)
# allcon=[0.0981566,
# 0.0368172,
# 0.0727857,
# 0.1094,
# 0.0213733,
# 0.176634,
# 0.107327,
# 0.196325,
# 0.122623,
# 0.186806,
# 0.0994387,
# 0.042993, 
# 0.0292858,
# 0.119824,
# 0.0794429,
# 0.106627,
# 0.138273,
# 0.0802485,
# 0.0605251,
# 0.142491,
# 0.139056,
# 0.157837,
# 0.103922,
# 0.0651167,
# 0.222909,
# 0.0436214,
# 0.0476045,
# 0.158086,
# 0.132768,
# 0.0305809,
# 0.087091,
# 0.265643,
# 0.112657,
# 0.150734,
# 0.12223,
# 0.197761,
# 0.0323147,
# 0.0791428,
# 0.0558404,
# 0.174463,
# 0.0970007,
# 0.189392,
# 0.263229,
# 0.212533,
# 0.143506,
# 0.172182,
# 0.0500165,
# 0.174847,
# 0.0971317,
# 0.353752,
# 0.121277,
# 0.0958203,
# 0.0627164,
# 0.153303,
# 0.0517064,
# 0.0875355,
# 0.0934111,
# 0.110788,
# 0.101663,
# 0.127262


# ]



# #allconSloop=np.transpose(allconSloop)
# allt1=[]
# allp=[]
# allp1=[]
# for i in range(100):

#      t1=stat.ttest_rel(allcon,allconSloop[i]).statistic
#      p1=stat.ttest_rel(allcon,allconSloop[i]).pvalue
#      p=stat.ttest_1samp(allconSloop[i],0.119430797).pvalue
#      allp.append(p.tolist())
#      allt1.append(t1.tolist())
#      allp1.append(p1.tolist())
# plt.hist(allp)
#plt.hist(allt1)

#pval=stat.ttest_1samp(allt, 0).pvalue     
  
# writing the data into the file 
#with file:     
#    write = csv.writer(file) 
#    write.writerow(allcon)  
#import scipy.stats
#p=scipy.stats.ttest_rel(allconS, allcon).pvalue
#
#pall.append(p)