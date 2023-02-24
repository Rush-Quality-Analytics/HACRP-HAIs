import pandas as pd
import numpy as np
import random
import sys
import os
import scipy as sc
import warnings
from scipy import stats
from scipy.stats import gmean
from scipy.stats.mstats import winsorize
from statsmodels.stats.proportion import proportion_confint
import matplotlib.pyplot as plt
from numpy import log10, sqrt

 
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = '/Users/kenlocey/GitHub/HACRP-HAIs/'

############################################################################################
##################### Changes in Penalty Assignment ########################################
############################################################################################

main_df = pd.read_pickle(mydir + 'data/yearly_compiled/HACRP-File-04-2022_HAI-File-04-2021.pkl')
print(list(main_df))

main_df = main_df[~main_df['Payment Reduction'].isin([np.nan, float('NaN')])]
main_df = main_df[~main_df['Total HAC Score'].isin([np.nan, float('NaN')])]
rumc_df = main_df[main_df['Facility ID'] == '140119']
rumc_df = rumc_df.filter(items=['Total HAC Score (derived)', 'Total HAC Score (SIS)', 
                                'Payment Reduction (derived)', 'Payment Reduction (SIS)', 
                                'Payment Reduction'], axis=1)

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

main_df['Avg device days'] = main_df[['CAUTI Volume', 'CLABSI Volume']].mean(axis=1)
main_df['Avg patient days'] = main_df[['MRSA Volume', 'CDI Volume']].mean(axis=1)

print('main_df.shape:', main_df.shape)

fig = plt.figure(figsize=(10, 18))
rows, cols = 4, 2
fs = 14
radius = 200

#########################################################################################
################################ GENERATE FIGURE ########################################
#########################################################################################

################################## SUBPLOT 1 ############################################
ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=2, rowspan=1)
x = main_df['Avg device days']+1
y = main_df['change in rank']
ax1.scatter(x, y,facecolors='none',
        s = 30, edgecolors='0.1', linewidths=1.)
ax1.scatter(x, y, c = 'w', s = 30, edgecolors='k', linewidths=0.0,
                )

c1 = 'r'
c2 = 'b'
c3 = '0.4'
s = 60

tdf = main_df[(main_df['Payment Reduction (SIS)'] == 'Yes') & (main_df['Payment Reduction'] == 'Yes')]
x1 = tdf['Avg device days']+1
y1 = tdf['change in rank']
plt.scatter(x1, y1, s=s, c=c3, edgecolors='w', linewidths=0.2)

tdf = main_df[(main_df['Payment Reduction'] == 'Yes') & (main_df['Payment Reduction (SIS)'] == 'No')]
x1 = tdf['Avg device days']+1
y1 = tdf['change in rank']
plt.scatter(x1, y1, s=s, c=c1, edgecolors='w', linewidths=0.2)

tdf = main_df[(main_df['Payment Reduction (SIS)'] == 'Yes') & (main_df['Payment Reduction'] == 'No')]
x1 = tdf['Avg device days']+1
y1 = tdf['change in rank']
plt.scatter(x1, y1, s=s, c=c2, edgecolors='w', linewidths=0.2)

plt.ylabel('worsened                improved', fontsize=fs)#, fontweight='bold')
plt.xlabel('Average device days', fontsize=fs, fontweight='bold')
plt.tick_params(axis='both', labelsize=fs-4)
plt.xlim(30, 1.1*max(x))
#plt.ylim(-1500, 1900)
plt.hlines(0, 0, max(x), colors='k')
plt.xscale('log')

plt.scatter([-10], [-10], s=80, c=c1, label='Only penalized via use of SIR')
plt.scatter([-10], [-10], s=80, c=c2, label='Only penalized via use of SIS')
plt.scatter([-10], [-10], s=80, c=c3, label='Penalized via SIR and SIS')
plt.legend(bbox_to_anchor=(-.01, 1.05, 1.02, .2), loc=10, ncol=2, frameon=True, mode="expand",
           prop={'size':fs, 'weight':'bold'})


####################### SUBPLOT 2 ##################################################################

ax2 = plt.subplot2grid((rows, cols), (1, 0), colspan=2, rowspan=1)

x = main_df['Avg patient days']+1
y = main_df['change in rank']
ax2.scatter(x, y,facecolors='none',
        s = 30, edgecolors='0.1', linewidths=1.)
ax2.scatter(x, y, c = 'w', s = 30, edgecolors='k', linewidths=0.0,
                )

tdf = main_df[(main_df['Payment Reduction (SIS)'] == 'Yes') & (main_df['Payment Reduction'] == 'Yes')]
x1 = tdf['Avg patient days']+1
y1 = tdf['change in rank']
plt.scatter(x1, y1, s=s, c=c3, edgecolors='w', linewidths=0.2)

tdf = main_df[(main_df['Payment Reduction'] == 'Yes') & (main_df['Payment Reduction (SIS)'] == 'No')]
x1 = tdf['Avg patient days']+1
y1 = tdf['change in rank']
plt.scatter(x1, y1, s=s, c=c1, edgecolors='w', linewidths=0.2)

tdf = main_df[(main_df['Payment Reduction (SIS)'] == 'Yes') & (main_df['Payment Reduction'] == 'No')]
x1 = tdf['Avg patient days']+1
y1 = tdf['change in rank']
plt.scatter(x1, y1, s=s, c=c2, edgecolors='w', linewidths=0.2)

plt.ylabel('worsened               improved', fontsize=fs)#, fontweight='bold')
plt.xlabel('Average patient days', fontsize=fs, fontweight='bold')
plt.tick_params(axis='both', labelsize=fs-4)
plt.xlim(1000, 1.1*max(x))
#plt.ylim(-1500, 2000)
plt.hlines(0, 0, max(x), colors='k')
#plt.text(-150, 0.6, 'CAUTI', fontsize=fs+3, fontweight='bold', rotation=90)
plt.xscale('log')
plt.text(280, 0, 'Change in rank after accounting for\n       random effects of volume', rotation=90,
         fontsize=fs+2, fontweight='bold')

plt.subplots_adjust(wspace=0.5, hspace=0.25)
plt.savefig(mydir+'/figures/change_in_rank.png', dpi=600, bbox_inches = "tight")
plt.close()




