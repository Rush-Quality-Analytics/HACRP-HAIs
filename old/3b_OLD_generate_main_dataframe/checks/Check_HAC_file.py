import pandas as pd
import numpy as np
import random
from math import pi
import sys
import os
import scipy as sc
import warnings
from scipy.stats import binned_statistic
from numpy import log10, sqrt
from scipy import stats
from scipy.stats.mstats import winsorize

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = '/Users/kenlocey/GitHub/HACRP-HAIs/'


hac_df = pd.read_pickle(mydir + "CareCompare_data/CombinedFiles_HACRP/Facility.pkl")
hac_df = hac_df[hac_df['file_month'] == '04']
hac_df = hac_df[hac_df['file_year'] == '2022']
hac_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
hac_df.dropna(how='all', axis=1, inplace=True)
hac_df.sort_values(by='Facility ID', inplace=True)
#print(list(set(hac_df['Facility ID'])))
print('hac_df.shape', hac_df.shape)
print(sorted(list(hac_df)), '\n')


hac_df_orig = pd.read_csv(mydir + "CareCompare_data/FY_2022_HAC_Reduction_Program_Hospital.csv", 
                          encoding = "ISO-8859-1", dtype={'Facility ID': 'string'})
hac_df_orig['Facility ID'] = hac_df_orig['Facility ID'].values.astype(str)
#print(list(set(hac_df_orig['Facility ID'])))
#sys.exit()

hac_df_orig.dropna(how='all', axis=1, inplace=True)
hac_df_orig.sort_values(by='Facility ID', inplace=True)
cols1 = ['PSI 90 End Date', 'PSI 90 Footnote', 'PSI 90 Start Date', 'PSI 90 W Z Score']
cols2 = ['PSI-90 End Date', 'PSI-90 Footnote', 'PSI-90 Start Date', 'PSI-90 W Z Score']
for i, col in enumerate(cols1):
    if col in list(hac_df_orig):
        hac_df_orig.rename(columns={col: cols2[i]}, inplace=True)
print('hac_df_orig.shape', hac_df_orig.shape)
print(sorted(list(hac_df_orig)), '\n')

if hac_df.shape == hac_df_orig.shape: 
    print("Subset of curated file has the same number of rows and columns as the original file")
else: 
    print("Subset of curated file DOES NOT have the same number of rows and columns as the original file")
    sys.exit()

if sorted(list(hac_df)) == sorted(list(hac_df_orig)): 
    print("Subset of curated file has the same column labels as the original file")
else: 
    print("Subset of curated file DOES NOT have the same column labels as the original file")
    sys.exit()

hac_df = hac_df.filter(items=list(hac_df_orig), axis=1)
print(hac_df.head(), '\n\n')
print(hac_df_orig.head())
