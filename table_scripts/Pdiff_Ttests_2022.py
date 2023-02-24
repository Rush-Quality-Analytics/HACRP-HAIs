import pandas as pd
import numpy as np
import random
import sys
import os
import scipy as sc
import warnings
from scipy import stats
from scipy.stats import gmean, gstd, iqr
from scipy.stats.mstats import winsorize
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
from numpy import log10, sqrt


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = '/Users/kenlocey/GitHub/HACRP-HAIs/'

############################################################################################
########################### Custom Functions ###############################################
############################################################################################

def p_diff(x, y):
    return np.round(100 * (np.abs(x - y)/np.mean([x, y])),1)

def get_central_tendency(x, metric):
    if np.min(x) == 0:
        x = x + 1
     
    if metric == 'gmean':
        avg = gmean(x)
        sd = gstd(x)
    
    elif metric == 'mean':
        avg = np.nanmean(x)
        sd = np.nanstd(x)
    
    elif metric == 'sqrt':
        avg = np.nanmean(np.sqrt(x))**2
        sd = np.nanstd(np.sqrt(x))**2
        
    elif metric == 'median':
        avg = np.nanmedian(x)
        sd = iqr(x)
    
    return avg, sd
    

############################################################################################
##################### Changes in Penalty Assignment ########################################
############################################################################################

main_df = pd.read_pickle(mydir + 'data/yearly_compiled/HACRP-File-04-2022_HAI-File-04-2021.pkl')
#main_df = main_df[~main_df['Payment Reduction'].isin([np.nan, float('NaN')])]
main_df = main_df[~main_df['Total HAC Score'].isin([np.nan, float('NaN')])]
print(list(main_df))
print('main_df.shape:', main_df.shape)


penalized_SIS = main_df['Payment Reduction (SIS)'].tolist()
penalized_SIR = main_df['Payment Reduction'].tolist()

fptnp = 0
fnptp = 0
no_change = 0
fptp = 0
fnptnp = 0
p_sir = 0
p_sis = 0
np_sir = 0
for i, v1 in enumerate(penalized_SIR):
    v2 = penalized_SIS[i]
    
    if v1 == 'Yes':
        p_sir += 1
    if v1 == 'No':
        np_sir += 1
    if v2 == 'Yes':
        p_sis += 1
    if v1 == 'Yes' and v2 == 'No':
        fptnp += 1
    elif v1 == 'No' and v2 == 'Yes':
        fnptp += 1
    elif v1 == 'Yes' and v2 == 'Yes':
        fptp += 1
    elif v1 == 'No' and v2 == 'No':
        fnptnp += 1
        
d = len(penalized_SIR)
print('\n')
print('No. of hospitals NOT penalized via SIR:', np_sir)
print('No. of hospitals penalized via SIR:', p_sir)
print('No. of hospitals penalized via SIS:', p_sis)
print('No. of hospitals penalized via SIR but not penalized via SIS:', fptnp, ',', np.round(100*fptnp/d,2), '%')
print('No. of hospitals penalized via SIS but not penalized via SIR:', fnptp, ',', np.round(100*fnptp/d,2), '%')
print('No. of hospitals penalized via both SIR and SIS:', fptp, ',', np.round(100*fptp/d,2), '%')
print('No. of hospitals not penalized via both SIR and SIS:', fnptnp, ',', np.round(100*fnptnp/d,2), '%', '\n')

avg_hac_score = np.nanmean(main_df['Total HAC Score'])
print('Mean total HAC score (using SIR):', avg_hac_score)
avg_hac_score = np.nanmean(main_df['Total HAC Score (SIS)'])
print('Mean total HAC score (using SIS):', avg_hac_score, '\n\n')


        
####################################################################################################        
################# GET VOLUMES ######################################################################
####################################################################################################

metrics = ['mean', 'sqrt', 'median', 'gmean']

for metric in metrics:
    print('---------------  ' + metric + '  ---------------\n')
    
    #################### SIR based ####################
    print('Volumes across hospitals (SIR-based penalties):\n') 
    
    hais = ['CAUTI', 'CLABSI', 'MRSA', 'CDI']
    for hai in hais:
        tdf = main_df[~main_df[hai + ' Volume'].isin([np.nan, float('NaN')])]
        tdf = tdf[tdf['Payment Reduction'] == 'No']
        avg_np, sd = get_central_tendency(tdf[hai + ' Volume'], metric)
        print(hai, '(non-penalized):', avg_np, ',', sd, 'SD')
        
        sir_df = main_df[main_df['Payment Reduction'] == 'Yes']
        sir_df = sir_df[~sir_df[hai + ' Volume'].isin([np.nan, float('NaN')])]
        avg_p, sd = get_central_tendency(sir_df[hai + ' Volume'], metric)
        
        s = '% difference from non-penalized'
        pd = str(p_diff(avg_p, avg_np)) + s
        
        x1 = tdf[hai + ' Volume']
        x2 = sir_df[hai + ' Volume']
        
        if metric == 'sqrt':
            x1 = np.sqrt(x1)
            x2 = np.sqrt(x2)
            
        if metric == 'gmean':
            x1 = np.log10(x1+1)
            x2 = np.log10(x2+1)
        
        if metric == 'median':
            stat, pval, m, table = stats.median_test(x1, x2,
                            nan_policy='omit',
                            )
        else:
            t_stat, pval = stats.ttest_ind(x1, x2,
                        equal_var=False, #nan_policy='omit',
                        alternative='less',
                        )
            
        print(hai, '(penalized) :', avg_p, ',', sd, ' | ', pd, ', p =', pval, '\n')
    
    #print('\n')
       
        
    #################### SIS based ####################    
    print('\n')
    print('Volumes across hospitals (SIS-based penalties):\n')
    
    for hai in hais:
        tdf = main_df[~main_df[hai + ' Volume'].isin([np.nan, float('NaN')])]
        tdf = tdf[tdf['Payment Reduction (SIS)'] == 'No']
        avg_np, sd = get_central_tendency(tdf[hai + ' Volume'], metric)
        print(hai, '(non-penalized):', avg_np, ',', sd, 'SD')
        
        sis_df = main_df[main_df['Payment Reduction (SIS)'] == 'Yes']
        sis_df = sis_df[~sis_df[hai + ' Volume'].isin([np.nan, float('NaN')])]
        avg_p, sd = get_central_tendency(sis_df[hai + ' Volume'], metric)
        
        s = '% difference from non-penalized'
        pd = str(p_diff(avg_p, avg_np)) + s
        
        x1 = tdf[hai + ' Volume']
        x2 = sis_df[hai + ' Volume']
        
        if metric == 'sqrt':
            x1 = np.sqrt(x1)
            x2 = np.sqrt(x2)
        
        if metric == 'median':
            stat, pval, m, table = stats.median_test(x1, x2,
                            nan_policy='omit',
                            #random_state=1,
                            #alternative='less',
                        )
        else:
            t_stat, pval = stats.ttest_ind(x1, x2,
                        equal_var=False, nan_policy='omit',
                        permutations=None, random_state=1,
                        #alternative='less',
                    )
        print(hai, '(penalized):', avg_p, ',', sd, ' | ', pd, ', p =', pval, '\n')
    
    print('\n')

