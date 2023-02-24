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

main_df = pd.read_pickle(mydir + 'data/Compiled_HCRIS-HACRP-RAND/Compiled_HCRIS-HACRP-RAND.pkl')
main_df = main_df[~main_df['Payment Reduction'].isin([np.nan, float('NaN')])]

main_df.rename(columns={
        'CAUTI Urinary Catheter Days': 'CAUTI Volume', 
        'CLABSI Number of Device Days': 'CLABSI Volume', 
        'MRSA patient days': 'MRSA Volume', 
        'CDI patient days': 'CDI Volume',
        }, inplace=True)

#print(list(main_df))
#sys.exit()
print('main_df.shape:', main_df.shape)


yrs = sorted(list(set(main_df['FILE_YEAR'])))
penalized_SIR = main_df['Payment Reduction'].tolist()

p_sir = 0
np_sir = 0
for i, v1 in enumerate(penalized_SIR):
    
    if v1 == 'Yes':
        p_sir += 1
    if v1 == 'No':
        np_sir += 1
        
d = len(penalized_SIR)
print('\n')
print('No. of hospitals NOT penalized via SIR:', np_sir)
print('No. of hospitals penalized via SIR:', p_sir)

        
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
        
        yrly = []
        for yr in yrs:
            print('YEAR:', yr, 'metric:', metric)
            tdf1 = tdf[tdf['FILE_YEAR'] == yr]
            
            tdf2 = tdf1[tdf1['Payment Reduction'] == 'No']
            avg_np, sd = get_central_tendency(tdf2[hai + ' Volume'], metric)
            print('    ', hai, '(non-penalized):', avg_np, ',', sd, 'SD')
            
            tdf3 = tdf1[tdf1['Payment Reduction'] == 'Yes']
            tdf3 = tdf3[~tdf3[hai + ' Volume'].isin([np.nan, float('NaN')])]
            avg_p, sd = get_central_tendency(tdf3[hai + ' Volume'], metric)
            
            s = '% difference from non-penalized'
            pdif = str(p_diff(avg_p, avg_np)) + s
            
            x1 = tdf2[hai + ' Volume']
            x2 = tdf3[hai + ' Volume']
            
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
                
            print('    ', hai, '(penalized) :', avg_p, ',', sd, ' | ', pdif, ', p =', pval, '\n')
            yrly.append(p_diff(avg_p, avg_np))
            
        print(hai, '| Average yearly % difference:', np.nanmean(yrly), ', SD =', np.nanstd(yrly))
        print('\n')

