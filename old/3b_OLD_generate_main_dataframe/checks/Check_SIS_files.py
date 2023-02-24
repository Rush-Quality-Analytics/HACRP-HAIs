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

mydir2 = '/Users/kenlocey/GitHub/HAIs/'
mydir = '/Users/kenlocey/GitHub/HACRP-HAIs/'


sis_df = pd.read_pickle(mydir + "optimize/optimized_by_quarter/CAUTI/CAUTI_Data_opt_for_SIRs_2015_04.pkl")
print(list(sis_df), '\n')
print(sis_df.head(), '\n')
sys.exit()

sis_df = pd.read_pickle(mydir2 + "data/WinsorizedZscores.pkl")
#print(sis_df.head(), '\n')
'''
dates = sis_df['file date'].tolist()

mos = []
yrs = []
for d in dates:
    mos.append(d[5:7])
    yrs.append(d[0:4])
    
sis_df['file_month'] = mos
sis_df['file_year'] = yrs
sis_df = sis_df[(sis_df['file_month'] == '04') & (sis_df['file_year'] == '2021')]
'''

sys.exit()

hosps1 = sis_df['Facility and File Date'].tolist()
print(hosps1[:10], '\n')

hosps2 = sis_df['provider'].tolist()
print(hosps2[:10], '\n')

r_hosps = []
for i, h in enumerate(hosps2):
    if h[-1] == '-':
        print(hosps1[i], ':', h)
        h = '0' + h[:-1]
        print(h)
        #sys.exit()
        
        
#hac_df = pd.read_pickle(mydir + "yearly_compiled/HACRP-File-01-2022_HAI-File-04-2021.pkl")
#print(list(hac_df), '\n')

