import pandas as pd
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt
from math import pi
import sys
import os
import scipy as sc
import warnings
from numpy import log10, sqrt
from scipy import stats
from mpl_toolkits.axes_grid.inset_locator import inset_axes

warnings.filterwarnings('ignore')
mydir = os.path.expanduser("~/GitHub/HACRP-HAIs/")

#########################################################################################
################################ FUNCTIONS ##############################################
#########################################################################################
def obs_pred_rsquare(obs, pred):
    print(np.min(obs), np.min(pred))
    if np.min(obs) < 0:
        obs = np.array(obs) - np.min(obs)
    if np.min(pred) < 0:
        pred = np.array(pred) - np.min(pred)
    print(np.min(obs), np.min(pred))
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)

def histogram_intersection(h1, h2):
    print(sum(h1), sum(h2))
    i1 = 100 * np.sum(np.minimum(np.array(h1)/sum(h1), np.array(h2)/sum(h2)))
    i2 = 100 * np.sum(np.minimum(np.array(h1), np.array(h2)))/sum(h2)
    return i1, i2
    

#########################################################################################
########################## IMPORT HAI DATA ##############################################
#########################################################################################

num_bins = 30

hi_CAUTI = []
hi_CLABSI = []
hi_CDIFF = []
hi_MRSA = []

fdates = ['2021_04']

for fdate in sorted(fdates, reverse=False):
    
    main_df = pd.read_pickle(mydir + 'data/yearly_compiled/HACRP-File-04-2022_HAI-File-04-2021.pkl')
    main_df = main_df[~main_df['Total HAC Score'].isin([np.nan, float('NaN')])]
    main_df = main_df.filter(items=['Total HAC Score', 'Total HAC Score (random)'], axis=1)
    print(list(main_df))
    print('main_df.shape:', main_df.shape)
    
    #########################################################################################
    ##################### DECLARE FIGURE 3A OBJECT ##########################################
    #########################################################################################

    fig = plt.figure(figsize=(6, 6))
    rows, cols = 1, 1
    fs = 14
    radius = 2

    #########################################################################################
    ################################ GENERATE FIGURE ########################################
    #########################################################################################

    ################################## SUBPLOT 1 ############################################
    ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)
    
    y = main_df['Total HAC Score']
    y = y.tolist()
    x = main_df['Total HAC Score (random)']
    x = x.tolist()
    x = sorted(x)
    y = sorted(y)

    min_x = min([min(x), min(y)])
    max_x = max([max(x), max(y)])

    counts2, bins2, bars2 = plt.hist(y, bins=np.linspace(min_x, max_x, num_bins), histtype='step', 
                                     density=False, color='b', 
                                     label='Total HAC scores (actual)', 
                                     linewidth=3, alpha=0.5)
    counts1, bins1, bars1 = plt.hist(x, bins=np.linspace(min_x, max_x, num_bins), histtype='step', 
                                     density=False, color='r', 
                                     label='Total HAC scores\n(with random expectations)',
                                     linewidth=3, alpha=0.5)
    
    hi, hi2 = histogram_intersection(counts1, counts2)
    print(hi, hi2)
    r2 = obs_pred_rsquare(counts2, counts1)
    print('histogram intersection:', np.rint(hi), ' r2:', np.rint(r2))
    
    plt.text(0.4*max_x, 0.94*max([max(counts1), max(counts2)]), '∩ = ' + str(np.round(hi,1)) + '%', fontsize=fs+7)
    
    s = r'$^{*}$' + ' Scores for CAUTI, CLABSI, MRSA and CDIFF based on\n'
    s += "  SIRs where reported no.'s of infections were replaced\n"
    s += '  with random outcomes.'
    
    plt.legend(bbox_to_anchor=(-0.03, 1.02, 1.05, .2), loc=10, ncol=1, frameon=True, mode="expand",prop={'size':fs+2})
    
    plt.ylabel('No. of hospitals', fontsize=fs+4, fontweight='bold')
    plt.xlabel('Total HAC score', fontsize=fs+4, fontweight='bold')
    plt.tick_params(axis='both', labelsize=fs-1)
    
    x = main_df['Total HAC Score (random)']
    y = main_df['Total HAC Score']
    minx = min([min(x), min(y)])
    maxx = max([max(x), max(y)])
    
    #########################################################################################
    ################################ FINAL FORMATTING #######################################
    #########################################################################################

    plt.subplots_adjust(wspace=0.5, hspace=0.4)
    plt.savefig(mydir+'/figures/Hists_HAC.png', dpi=400, bbox_inches = "tight")
    plt.close()
