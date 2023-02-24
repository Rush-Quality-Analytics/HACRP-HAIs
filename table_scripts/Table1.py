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

'''
rumc_df = main_df[main_df['Facility ID'] == '140119']
rumc_df = rumc_df.filter(items=['Total HAC Score (derived)', 'Total HAC Score (SIS)', 
                                'Payment Reduction (derived)', 'Payment Reduction (SIS)', 
                                'Payment Reduction', 'CAUTI Volume', 'CLABSI Volume', 
                                'MRSA Volume', 'CDI Volume'], axis=1)
print('RUMC:')
print(rumc_df, '\n')
'''

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

metrics = ['mean', 'gmean', 'median']

for metric in metrics:
    print('---------------  ' + metric + '  ---------------\n')
    
    #################### SIR based ####################
    print('Volumes across penalized hospitals (SIR-based):\n') 
    
    hais = ['CAUTI', 'CLABSI', 'MRSA', 'CDI']
    avg_ls = []
    for hai in hais:
        tdf = main_df[~main_df[hai + ' Volume'].isin([np.nan, float('NaN')])]
        tdf = tdf[tdf['Payment Reduction'] == 'No']
        avg, sd = get_central_tendency(tdf[hai + ' Volume'], metric)
        print(hai, ':', avg, ',', sd)
        avg_ls.append(avg)
    
    print('\n')
    s = '% difference from non-penalized'
    sir_df = main_df[main_df['Payment Reduction'] == 'Yes']
    for i, hai in enumerate(hais):
        tdf = sir_df[~sir_df[hai + ' Volume'].isin([np.nan, float('NaN')])]
        avg, sd = get_central_tendency(tdf[hai + ' Volume'], metric)
        
        pd = str(p_diff(avg, avg_ls[i])) + s
        print(hai, ':', avg, ',', sd, ' | ', pd)
        
        
    #################### SIS based ####################    
    print('\n')
    print('Volumes across penalized hospitals (SIS-based):\n')
    
    avg_ls = []
    for hai in hais:
        tdf = main_df[~main_df[hai + ' Volume'].isin([np.nan, float('NaN')])]
        tdf = tdf[tdf['Payment Reduction (SIS)'] == 'No']
        avg, sd = get_central_tendency(tdf[hai + ' Volume'], metric)
        print(hai, ':', avg, ',', sd)
        avg_ls.append(avg)
    
    print('\n')    
    s = '% difference from non-penalized'
    sis_df = main_df[main_df['Payment Reduction (SIS)'] == 'Yes']
    for i, hai in enumerate(hais):
        tdf = sis_df[~sis_df[hai + ' Volume'].isin([np.nan, 0, float('NaN')])]
        avg, sd = get_central_tendency(tdf[hai + ' Volume'], metric)
        
        pd = str(p_diff(avg, avg_ls[i])) + s
        print(hai, ':', avg, ',', sd, ' | ', pd)
        
    print('\n')

sys.exit()



'''
tdf = main_df[(main_df['Payment Reduction'] == 'Yes') & (main_df['Payment Reduction (SIS)'] == 'No')]
print('Mean days for hospitals penalized via SIR but not penalized via SIS:')
print('CAUTI:', np.nanmean(tdf['CAUTI Volume']))
print('CLABSI:', np.nanmean(tdf['CLABSI Volume']))
print('MRSA:', np.nanmean(tdf['MRSA Volume']))
print('CDIF:', np.nanmean(tdf['CDI Volume']))
print('\n')

tdf = main_df[(main_df['Payment Reduction'] == 'No') & (main_df['Payment Reduction (SIS)'] == 'Yes')]
print('Mean days for hospitals penalized via SIS but not penalized via SIR:')
print('CAUTI:', np.nanmean(tdf['CAUTI Volume']))
print('CLABSI:', np.nanmean(tdf['CLABSI Volume']))
print('MRSA:', np.nanmean(tdf['MRSA Volume']))
print('CDIF:', np.nanmean(tdf['CDI Volume']))
print('\n')

HAC_scores_SIR = main_df['Total HAC Score'].tolist()
HAC_scores_SIS = main_df['Total HAC Score (SIS)'].tolist()

best_HAC_score_SIR = np.nanmin(main_df['Total HAC Score'])
best_HAC_score_SIS = np.nanmin(main_df['Total HAC Score (SIS)'])

num_best_HAC_score_SIR = HAC_scores_SIR.count(best_HAC_score_SIR)
num_best_HAC_score_SIS = HAC_scores_SIS.count(best_HAC_score_SIS)

print('Number of hospitals with the best total HAC score (SIR):', num_best_HAC_score_SIR)
print('Number of hospitals with the best total HAC score (SIS):', num_best_HAC_score_SIS)

main_df['Rank in total HAC score (SIR)'] = main_df['Total HAC Score'].rank(axis=0, 
                                                                           method='min',
                                                                           na_option='keep',
                                                                           ascending=True)

main_df['Rank in total HAC score (SIS)'] = main_df['Total HAC Score (SIS)'].rank(axis=0, 
                                                                           method='min',
                                                                           na_option='keep',
                                                                           ascending=True)


main_df['change in rank'] = main_df['Rank in total HAC score (SIR)'] - main_df['Rank in total HAC score (SIS)']

# When changes in rank are positive, a hospital's rank improved
# When changes in rank are negative, a hospital's rank worsened.
'''




