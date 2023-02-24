import pandas as pd
import numpy as np
import random
import sys
import os
import scipy as sc
import warnings
from scipy import stats
import matplotlib
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = '/Users/kenlocey/GitHub/HACRP-HAIs/'



#############################################################################################
###################### GENERATE RESULTS FOR 2022 ############################################
#############################################################################################

main_df = pd.read_pickle(mydir + 'data/yearly_compiled/HACRP-File-04-2022_HAI-File-04-2021.pkl')

main_df = main_df[~main_df['Payment Reduction'].isin([np.nan, float('NaN')])]
main_df = main_df[~main_df['Total HAC Score'].isin([np.nan, float('NaN')])]
main_df['Rank in total HAC score (SIR)'] = main_df['Total HAC Score'].rank(axis=0, method='min', na_option='keep', ascending=True)
main_df['Rank in total HAC score (SIS)'] = main_df['Total HAC Score (SIS)'].rank(axis=0, method='min', na_option='keep', ascending=True)

main_df['change in rank'] = main_df['Rank in total HAC score (SIR)'] - main_df['Rank in total HAC score (SIS)']
# When changes in rank are positive, a hospital's rank improved
# When changes in rank are negative, a hospital's rank worsened.

main_df['Total patient days'] = main_df[['MRSA Volume', 'CDI Volume']].sum(axis=1)
main_df['Total device days'] = main_df[['CAUTI Volume', 'CLABSI Volume']].sum(axis=1)

###################### Cost report data ####################################################
cr_df = pd.read_pickle('~/GitHub/HACRP-HAIs/data/Compiled_HCRIS-HACRP-RAND/Compiled_HCRIS-HACRP-RAND.pkl')
cr_df = cr_df[cr_df['FILE_YEAR'] == '2022']
cr_df.drop(labels=['Total HAC Footnote', 'Total HAC Score', 'Total device days',
                           'HAI Measures End Date', 'HAI Measures Start Date',
                           'Payment Reduction Footnote',], axis=1, inplace=True)

cr_df.rename(columns={"PRVDR_NUM": "Facility ID"}, inplace=True)

h1 = cr_df['Facility ID'].tolist()
cr_df = cr_df[~cr_df['Facility ID'].isin([np.nan, float("NaN")])]
h2 = []

for h in h1:
    h = str(h)
    if len(h) < 6:
        h = '0' + h
        h2.append(h)
    else:
        h2.append(h)

cr_df['Facility ID'] = h2
main_df = main_df.merge(cr_df, on=['Facility ID'], how='outer')
main_df = main_df[~main_df['Payment Reduction'].isin([np.nan, float('NaN')])]
main_df = main_df[~main_df['Total HAC Score'].isin([np.nan, float('NaN')])]
print('main_df.shape:', main_df.shape, '\n')


####################################################################################################
##########  Hospitals that were penalized in 2022 but should not have been  ########################
####################################################################################################

print('Hospitals that were penalized:')
tdf1 = main_df[main_df['Payment Reduction'] == 'Yes']
print(tdf1.shape[0], 'hospitals')

print('Hospitals that were penalized but should not have been:')
tdf1 = main_df[(main_df['Payment Reduction'] == 'Yes') & (main_df['Payment Reduction (SIS)'] == 'No')]
print(tdf1.shape[0], 'hospitals')

tdf1 = tdf1[~tdf1["HAC penalty, final"].isin([np.nan, float('NaN')])] 
print(tdf1.shape[0], 'with payment data')

tdf1['HAC payment, no penalty'] = tdf1["HAC penalty, final"]/0.01
tdf1['HAC payment, w/ penalty'] = tdf1['HAC payment, no penalty'] - tdf1["HAC penalty, final"]

n = "${:,.2f}".format(np.round(np.sum(tdf1['HAC payment, w/ penalty']), 2))
print('    payment total (with penalties):', n)


n = "${:,.2f}".format(np.sum(tdf1['HAC payment, no penalty']))
print('    payment total (no penalties):', n)

tdf1['delta'] = tdf1['HAC payment, no penalty'] - tdf1['HAC payment, w/ penalty']
delta1 = np.sum(tdf1['delta'])
n = "${:,.2f}".format(np.round(delta1,2))
print('Gain by hospitals:', n)
n = "${:,.2f}".format(np.round(np.max(tdf1['delta']), 2))
print('Largest single gain:', n, '\n')

tdf1.sort_values(by='delta', inplace=True, ascending=False)



####################################################################################################
##########  Hospitals that were not penalized in 2022 but should have been  ########################
####################################################################################################

print('Hospitals that were not penalized:')
tdf = main_df[main_df['Payment Reduction'] == 'No']
print(tdf.shape[0], 'hospitals')

print('Hospitals that were not penalized but should have been:')
tdf = main_df[(main_df['Payment Reduction'] == 'No') & (main_df['Payment Reduction (SIS)'] == 'Yes')]
print(tdf.shape[0], 'hospitals')

tdf = tdf[~tdf["HAC penalty, final"].isin([np.nan, float('NaN')])] 
print(tdf.shape[0], 'hospitals with payment data')

tdf['HAC payment, no penalty'] = tdf["HAC penalty, final"]/0.01
tdf['HAC payment, w/ penalty'] = tdf['HAC payment, no penalty'] - tdf["HAC penalty, final"]
tdf['delta'] = tdf['HAC payment, no penalty'] - tdf['HAC payment, w/ penalty']

print('    payment total (with penalties):', "${:,.2f}".format(np.round(np.sum(tdf['HAC payment, w/ penalty']), 2)))
print('    payment total (no penalties):', "${:,.2f}".format(np.sum(tdf['HAC payment, no penalty'])))

delta2 = np.sum(tdf['delta'])
print('Loss by hospitals:', "${:,.2f}".format(np.round(delta2,2)))
print('Largest single loss:', "${:,.2f}".format(np.round(np.min(tdf['delta']),2)), '\n')
print('Net gain by hospitals:', "${:,.2f}".format(np.round(delta1 - delta2, 2)),'\n')


#########################################################################################
#######################   TRENDS ACROSS YEARS   #########################################
#########################################################################################

T_bad_pen_2022 = delta1
T_bad_pen = delta1*8/10**6
T_ok_pen_2022  = delta2
T_ok_pen = delta2*8/10**6


df = pd.read_pickle(mydir + '5_generate_penalty_df/expected_penalty_df.pkl')
yrs_pen = df['year']

avg_T_bad_pen = df['cum_sum_ia']/10**6
std_T_bad_pen = (df['cum_sum_std_ia']/10**6) * 2
Avgs_bad_pen = df['yrly_sum_ia'].astype(float)/10**6
Stds_bad_pen = (df['yrly_std_ia']/10**6) #* 2
             
avg_T_ok_pen = df['cum_sum_a']/10**6
std_T_ok_pen = (df['cum_sum_std_a']/10**6) * 2
Avgs_ok_pen = df['yrly_sum_a'].astype(float)/10**6
Stds_ok_pen = (df['yrly_std_a']/10**6) #* 2

avg_T_s = df['cum_sum_s']/10**6
std_T_s = (df['cum_sum_std_s']/10**6) * 2
Avgs_s = df['yrly_sum_s'].astype(float)/10**6
Stds_s = (df['yrly_std_s']/10**6) #* 2

##################### DECLARE FIGURE 3A OBJECT ##########################################

fig = plt.figure(figsize=(10, 8))
rows, cols = 2, 2
fs = 14
radius = 2

################################ GENERATE FIGURE ########################################

################################## Subplot 1 ############################################

ax1 = plt.subplot2grid((rows, cols), (0, 0), colspan=1, rowspan=1)

ln1 = ax1.errorbar(yrs_pen, Avgs_bad_pen, yerr= Stds_bad_pen, fmt='o', c='r', 
             #markersize=20, 
             mec='r',
             mfc='w',
             label='Biased penalties',
             zorder=1)
ln2 = ax1.scatter([2022], [T_bad_pen_2022/10**6], c='r', 
            s=80, 
            #label='Estimated',
            edgecolors='lightcoral',
            zorder=2)

ln1 = ax1.errorbar(yrs_pen, Avgs_ok_pen, yerr= Stds_ok_pen, fmt='o', c='b', 
             #markersize=20, 
             mec='b',
             mfc='w',
             label='Unbiased penalties',
             zorder=1)
ln2 = ax1.scatter([2022], [T_ok_pen_2022/10**6], c='b', 
            s=80, 
            #label='Estimated',
            edgecolors='steelblue',
            zorder=2)

plt.ylabel('Yearly penalties', fontsize=fs+4)
plt.xlabel('Year', fontsize=fs+4)
plt.tick_params(axis='both', labelsize=fs-1)
ax1.set_yticks([20, 40, 60, 80, 100])
ax1.set_yticklabels(['$20M', '$40M', '$60M', '$80M', '$100M'])

plt.legend(bbox_to_anchor=(-0.04, 1.02, 2.53, .2), loc=10, ncol=2, frameon=True, 
           handletextpad = -0.25,
           mode="expand",prop={'size':fs+2})


################################## Subplot 2 ############################################

ax2 = plt.subplot2grid((rows, cols), (0, 1), colspan=1, rowspan=1)

ln1 = ax2.errorbar(yrs_pen, avg_T_bad_pen, yerr=std_T_bad_pen, fmt='o', c='r', 
             #markersize=10, 
             mec='r',
             mfc='w',
             label='Derived estimate',
             zorder=1)
ln2 = ax2.scatter([2022], [T_bad_pen], c='r', 
            s=60, 
            edgecolors='lightcoral',
            label='Direct extrapolation',
            zorder=2)

ln1 = ax2.errorbar(yrs_pen, avg_T_ok_pen, yerr=std_T_ok_pen, fmt='o', c='b', 
             #markersize=10, 
             mec='b',
             mfc='w',
             label='Derived estimate',
             zorder=1)
ln2 = ax2.scatter([2022], [T_ok_pen], c='b', 
            s=60, 
            edgecolors='steelblue',
            label='Direct extrapolation',
            zorder=2)

plt.ylabel('Cumulative penalties', fontsize=fs+4)
plt.xlabel('Year', fontsize=fs+4)
plt.tick_params(axis='both', labelsize=fs-1)
ax2.set_yticks([100, 200, 300, 400, 500, 600, 700])
ax2.set_yticklabels(['$100M', '$200M', '$300M', '$400M', '$500M', '$600M', '$700M'])

'''
################################## Subplot 3 ############################################

ax3 = plt.subplot2grid((rows, cols), (1, 0), colspan=1, rowspan=1)
ln1 = ax3.errorbar(yrs_pen, Avgs_s, yerr= Stds_s, fmt='o', c='k', 
             #markersize=20, 
             mec='k',
             mfc='w',
             label='Derived estimate',
             zorder=1)
ln2 = ax3.scatter([2022], [T_s_2022/10**6], c='0.4', 
            s=80, 
            label='Estimated',
            edgecolors='0.7',
            zorder=2)

plt.ylabel('Yearly savings', fontsize=fs+2)
plt.xlabel('Year', fontsize=fs+2)
plt.tick_params(axis='both', labelsize=fs-1)
ax3.set_yticks([50, 60, 70, 80])
ax3.set_yticklabels(['$50M', '$60M', '$70M', '$80M'])


################################## Subplot 4 ############################################

ax4 = plt.subplot2grid((rows, cols), (1, 1), colspan=1, rowspan=1)

ln1 = ax4.errorbar(yrs_pen, avg_T_s, yerr=std_T_s, fmt='o', c='k', 
             #markersize=10, 
             mec='k',
             mfc='w',
             label='Derived estimate',
             zorder=1)
#ln2 = ax4.scatter([2022], [T_s], c='0.4', 
#            s=80, 
#            edgecolors='0.7',
#            label='Direct extrapolation',
#            zorder=2)

plt.ylabel('Cumulative savings', fontsize=fs+2)
plt.xlabel('Year', fontsize=fs+2)
plt.tick_params(axis='both', labelsize=fs-1)
ax4.set_yticks([100, 200, 300, 400, 500])
ax4.set_yticklabels(['$100M', '$200M', '$300M', '$400M', '$500M'])
'''

################################ FINAL FORMATTING #######################################
#########################################################################################

plt.subplots_adjust(wspace=0.45, hspace=0.)
plt.savefig(mydir+'/figures/expected_penalties.png', dpi=400, bbox_inches = "tight")
plt.close()