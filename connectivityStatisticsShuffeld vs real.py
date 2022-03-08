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
import mne


dfreal = pd.read_excel("C:/Users/yaara/Documents/morher-infant behavioral-neural synchrony/Free-Interaction/Allmeans shuffels64comb.xlsx", sheet_name ='real')
dfshuffle = pd.read_excel("C:/Users/yaara/Documents/morher-infant behavioral-neural synchrony/Free-Interaction/Allmeans shuffels64comb.xlsx", sheet_name ='shuffled')
wilcoall=[]
pvalall=[]
for x in range (1,65):

        pval=stat.wilcoxon(dfreal.iloc[:, x],dfshuffle.iloc[:, x]).pvalue
       # pval=stat.ttest_rel(dfreal.ix[:, x],dfshuffle.ix[:, x]).pvalue
        statW=stat.wilcoxon(dfreal.iloc[:, x],dfshuffle.iloc[:, x]).statistic
        wilcoall.append(statW)
        pvalall.append(pval)


PFDR= mne.stats.fdr_correction(pvalall, alpha=0.05)
#PBonferroni=mne.stats.bonferroni_correction(pvalall, alpha=0.05)