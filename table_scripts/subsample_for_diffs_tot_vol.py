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
import scikit_posthocs as sp

np.random.seed(100)

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = '/Users/kenlocey/GitHub/HACRP-HAIs/'

 
####################################################################################################        
################# Analyze differences in VOLUMES ###################################################
####################################################################################################

main_df = pd.read_pickle(mydir + 'data/yearly_compiled/HACRP-File-04-2022_HAI-File-04-2021.pkl')
main_df = main_df[~main_df['Total HAC Score'].isin([np.nan, float('NaN')])]
#main_df['total_volume'] = np.log10(main_df['CAUTI Volume']) + np.log10(main_df['CLABSI Volume']) + np.log10(main_df[['MRSA Volume', 'CDI Volume']].mean(axis=1))
main_df['total_volume'] = np.sqrt(main_df['CAUTI Volume']) + np.sqrt(main_df['CLABSI Volume']) + np.sqrt(main_df[['MRSA Volume', 'CDI Volume']].mean(axis=1))

sigdig = 5
    
not_pen_by_sir = main_df[main_df['Payment Reduction'] == 'No']
pen_by_sir = main_df[main_df['Payment Reduction'] == 'Yes']
    
not_pen_by_sis = main_df[main_df['Payment Reduction (SIS)'] == 'No']
pen_by_sis = main_df[main_df['Payment Reduction (SIS)'] == 'Yes']
never_pen = main_df[(main_df['Payment Reduction'] == 'No') & (main_df['Payment Reduction (SIS)'] == 'No')]
    
tdf1 = not_pen_by_sir[~not_pen_by_sir['total_volume'].isin([np.nan, float('NaN')])]
tdf2 = pen_by_sir[~pen_by_sir['total_volume'].isin([np.nan, float('NaN')])]
        
x1 = tdf1['total_volume'].tolist()
x2 = tdf2['total_volume'].tolist()        
t_stat, pval = stats.ttest_ind(x1, x2, 
                    equal_var=False, #nan_policy='omit',
                    permutations=None, random_state=1,
                    alternative='less',
                    )
print('\n')
print('One-sided t-tests:')
print('Not penalized by SIR vs. penalized by SIR:  t =', np.round(t_stat,sigdig), '| p = ', np.round(pval,sigdig))

tdf3 = not_pen_by_sis[~not_pen_by_sis['total_volume'].isin([np.nan, float('NaN')])]
tdf4 = pen_by_sis[~pen_by_sis['total_volume'].isin([np.nan, 0, float('NaN')])]

x3 = tdf3['total_volume'].tolist()
x4 = tdf4['total_volume'].tolist()       
t_stat, pval = stats.ttest_ind(x3, x4,
                equal_var=False, #nan_policy='omit',
                permutations=None, random_state=1,
                alternative='less',
            )

print('Not penalized by SIS vs. penalized by SIS:  t =', np.round(t_stat,sigdig), '| p = ', np.round(pval,sigdig))
print('\n')


################## Iterative Kruskal Wallis test ######### 
    
pen_by_both = main_df[(main_df['Payment Reduction'] == 'Yes') & (main_df['Payment Reduction (SIS)'] == 'Yes')]
pen_only_by_sir = main_df[(main_df['Payment Reduction'] == 'Yes') & (main_df['Payment Reduction (SIS)'] == 'No')]
pen_only_by_sis = main_df[(main_df['Payment Reduction'] == 'No') & (main_df['Payment Reduction (SIS)'] == 'Yes')]

ct = 0
pvals1 = [] # Kruskal Wallis 
Hs = [] # Kruskal Wallis
pvals2 = [] # Dunnet (MC): Never penalized vs. Penalized by SIR
pvals3 = [] # Dunnet (MC): Never penalized vs. Penalized by SIS
pvals4 = [] # Dunnet (MC): Penalized by SIR vs. Penalized by SIS

while ct < 10**2:
    ct += 1
    pen_both1 = pen_by_both.sample(frac=0.5, replace=False, random_state=ct)
    ids = pen_both1['Facility ID'].tolist()
    pen_both2 = pen_by_both[~pen_by_both['Facility ID'].isin(ids)]
        
    pen_by_SIR = pd.concat([pen_only_by_sir, pen_both1])
    pen_by_SIR.drop_duplicates(inplace=True)
    
    pen_by_SIS = pd.concat([pen_only_by_sis, pen_both2])
    pen_by_SIS.drop_duplicates(inplace=True)
    
    tdf5 = never_pen[~never_pen['total_volume'].isin([np.nan, float('NaN')])]
    tdf6 = pen_by_SIR[~pen_by_SIR['total_volume'].isin([np.nan, float('NaN')])]
    tdf7 = pen_by_SIS[~pen_by_SIS['total_volume'].isin([np.nan, float('NaN')])]
            
    x5 = tdf5['total_volume']
    x6 = tdf6['total_volume']
    x7 = tdf7['total_volume']
    
    #print(len(x5), len(x6), len(x7))
    #sys.exit()
    minlen = min([len(x5), len(x6), len(x7)])
    n = 100
    x5 = np.random.choice(x5, size=minlen, replace=False)
    x6 = np.random.choice(x6, size=minlen, replace=False)
    x7 = np.random.choice(x7, size=minlen, replace=False)
    
    H, pval = stats.kruskal(x5, x6, x7)
    
    Hs.append(H)
    pvals1.append(pval)
    r = sp.posthoc_dunn([x5, x6, x7], p_adjust = 'simes-hochberg')
    #r = sp.posthoc_dunn([x5, x6, x7], p_adjust = 'bonferroni')
    #r = sp.posthoc_dunn([x5, x6, x7], p_adjust = 'fdr_tsbky')
    
    pvals2.append(r[1][2])
    pvals3.append(r[1][3])    
    pvals4.append(r[2][3])   

print('3-sample Kruskal-Wallis test followed by Dunnett multiple comparisons test')
print('[10K iterations of random sampling to achieve independent observations and equal samples sizes]\n')
print('Never penalized vs. Penalized by SIR vs Penalized by SIS:')
print('Kruskal Wallace (avg H) = ', np.round(np.nanmean(Hs), sigdig), ', SD =', np.round(np.nanstd(Hs), sigdig), ' | avg p = ', np.round(np.nanmean(pvals1), sigdig), ', SD =', np.round(np.nanstd(pvals1), sigdig))

print('\n')
print('Dunnett multiple-comparisons test')
print('Never penalized vs. Penalized by SIR: avg p = ', np.round(np.nanmean(pvals2), sigdig), ', SD =', np.round(np.nanstd(pvals2), sigdig))
print('Never penalized vs. Penalized by SIS: avg p = ', np.round(np.nanmean(pvals3), sigdig), ', SD =', np.round(np.nanstd(pvals3), sigdig))   
print('Penalized by SIR vs. Penalized by SIS: avg p = ', np.round(np.nanmean(pvals4), sigdig), ', SD =', np.round(np.nanstd(pvals4), sigdig))   
print('\n\n')



