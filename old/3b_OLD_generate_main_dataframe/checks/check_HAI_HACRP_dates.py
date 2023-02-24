import pandas as pd
import numpy as np
import random
import sys
import os
import scipy as sc
import warnings
from scipy import stats
from scipy.stats.mstats import winsorize
from statsmodels.stats.proportion import proportion_confint


pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = '/Users/kenlocey/GitHub/HACRP-HAIs/'

#hac_mos = ['04', '01']
#hac_yrs = ['2022', '2022']
          
#mos = ['04', '04']
#yrs = ['2021', '2021']

#hac_mo = '10'
#hac_yr = '2021'
hac_df = pd.read_pickle(mydir + "data/CareCompare_data/CombinedFiles_HACRP/Facility.pkl")

hac_months = list(set(hac_df['file_month'].tolist()))
hac_years = list(set(hac_df['file_year'].tolist()))

for yr in hac_years:
    df1 = hac_df[hac_df['file_year'] == yr]
    for mo in hac_months:
        df2 = df1[df1['file_month'] == mo]
        if df2.shape[0] == 0:
            continue
        else:
            df2.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
            df2.dropna(how='all', axis=1, inplace=True)
            df2 = df2[~df2['Total HAC Score'].isin([np.nan, float('NaN')])]
            df2.sort_values(by='Facility ID', inplace=True)
            hac_hosps = list(set(df2['Facility ID'].tolist()))
            #print(len(hac_hosps), 'hospitals in HACRP file')
            #print('hac_df.shape', df2.shape[0])

            try:
                start_date = list(set(df2['HAI Measures Start Date'].tolist()))
                start_date = start_date[0]
                
                end_date = list(set(df2['HAI Measures End Date'].tolist()))
                end_date = end_date[0]
                

            except:
                start_date = list(set(df2['Domain 2 Start Date'].tolist()))
                start_date = start_date[0]
                
                end_date = list(set(df2['Domain 2 End Date'].tolist()))
                end_date = end_date[0]
    
            #########################################################################################
            #############################   CAUTI   #################################################
            #########################################################################################

            cauti_df = pd.read_pickle(mydir + "1_preprocess_CareCompare_data/preprocessed_HAI_data/CAUTI_Data.pkl")
            cauti_df = cauti_df[cauti_df['Start Date'] == start_date]
            cauti_df = cauti_df[cauti_df['End Date'] == end_date]
            if cauti_df.shape[0] > 0:
                print('HAC file year/mo:', yr, '/', mo)
                print('HAI Measures Start Date:', start_date)
                print('HAI Measures End Date:', end_date)
                print('cauti_df.shape:', cauti_df.shape, '\n')
                #sys.exit()


