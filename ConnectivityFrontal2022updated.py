# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 13:04:32 2022

@author: yaara
"""

import os
import mne
import pandas as pd
import numpy as np
from mne.connectivity import spectral_connectivity as sc

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
PATH_DIR_DEST = "C:/Users/yaara/Documents/morher-infant behavioral-neural synchrony/Free-Interaction/connectivity4to7hzwpli/"
PATH_DURATION_CSV = "C:/Users/yaara/Documents/morher-infant behavioral-neural synchrony/Free-Interaction/freeint.xlsx"
SBJ_COL_NAME = "file"
DUR_COL_NAME = "EPOCHS NUM"
OFFSET_START = 10
OFFSET_END = 10
TAG = 'F2F'

# -----------------------------------------------------------------------------
# USER ARGUMENTS VALIDATION
# -----------------------------------------------------------------------------
assert os.path.isdir(PATH_DIR_SRC), "invalid source folder"
assert os.path.isdir(PATH_DIR_DEST), "invalid destination folder"
assert os.path.isfile(PATH_DURATION_CSV), "invalid path to csv file"
assert PATH_DURATION_CSV.endswith(".xlsx"), "invalid file type"
excel_file = pd.read_excel(PATH_DURATION_CSV)
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

# loop over every subject pair
for (subj_1, subj_2) in files_ls:
    # load and pre-process data
    os.chdir(PATH_DIR_SRC)
    raw_1 = mne.read_epochs(subj_1, preload=True)
    raw_2 = mne.read_epochs(subj_2, preload=True)
    raw_1.pick_types(eeg=True,
                     exclude=['FCz', 'Fp1', 'Fp2', 'O1', 'O2', 'Oz', 
                              'Cz', 'Pz', 'Fz'])
    raw_2.pick_types(eeg=True,
                     exclude=['Fp1', 'Fp2', 'O1', 'O2', 'Oz', 'Cz',
                              'Pz', 'Fz'])
    apply_offset(raw_1, OFFSET_START, OFFSET_END, dic_duration[subj_1[0:6]])
    synch_drop_logs(raw_1, raw_2)
    drop_comb = zip(raw_1.drop_log, raw_2.drop_log)

    # preparations for spectral connectivity
    data_1 = raw_1.get_data()
    data_2 = raw_2.get_data()
    data_combined = np.concatenate((data_1, data_2), axis=1)
    info = mne.create_info(
        ch_names=raw_1.info['ch_names']+raw_1.info['ch_names'],
        ch_types=np.repeat('eeg', 16), sfreq=raw_1.info['sfreq'])
    events = np.array([np.array([i, 0, i])
                       for i in range(data_combined.shape[0])])
    events_id = dict(zip([str(x[0]) for x in events], [x[0] for x in events]))
    raw_combined = mne.EpochsArray(data_combined, info, events, -0.5,
                                   events_id)
   # raw_combined.plot()
    raw_combined.info['chs'] = raw_1.info['chs']+raw_2.info['chs']
    freq_bands = {'theta': (4, 7),
                  'alpha': (8, 12),
                  'beta': (13, 20),
                  'gamma': (32, 40)}
    fmin = np.array([f for f, _ in freq_bands.values()])
    fmax = np.array([f for _, f in freq_bands.values()])
   # indices = (np.array(range(11)), np.array(range(11, 22)))

    # spectral connectivity
    sc_data = sc(raw_combined, method='pli', fmin=fmin, fmax=fmax,
                 mode='multitaper', faverage=True, n_jobs=1, verbose=False)
    connB, freqsB, timesB, n_epochsB, n_tapersB = sc_data

    # export as .csv file
    os.chdir(PATH_DIR_DEST)
    file_name = subj_1[0:6]+TAG+'.csv'
    chnl_1 = np.array(list(range(16)) * 16)
    chnl_2 = chnl_1.copy()
    chnl_1.sort()
    pd.DataFrame(
        dict(channel_1=chnl_1, channel_2=chnl_2,
             **{key: np.reshape(connB[:, :, idx], 16 ** 2)
                for idx, key in enumerate(freq_bands.keys())})
            ).to_csv(file_name)