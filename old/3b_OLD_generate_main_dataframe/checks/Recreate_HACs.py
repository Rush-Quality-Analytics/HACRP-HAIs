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

#########################################################################################
#######################   CUSTOM FXNS   #################################################
#########################################################################################

def Winsorize_it(x):
    X = []
    for i in x:
        if np.isnan(i) == False:
            X.append(i)
        
    p5 = np.percentile(np.array(X), 5)
    p95 = np.percentile(np.array(X), 95)
    
    WinScores = []
    for i in x:
        if np.isnan(i) == False: 
            if i >= p5 and i <= p95:
                WinScores.append(i)
            elif i < p5:
                WinScores.append(p5)
            elif i > p95:
                WinScores.append(p95)
        else:
            WinScores.append(np.nan)
            
    return WinScores
        

def ZScore_it(x):
    zscores = (x - np.nanmean(x)) / np.nanstd(x)
    return zscores


def err(T, E):
    e = np.abs([E - T])
    return e


#########################################################################################
####################   LOAD HACRP AND HAI DATA   ########################################
#########################################################################################

HAC_df = pd.read_pickle(mydir + "data/CareCompare_data/CombinedFiles_HACRP/Facility.pkl")
HAC_df = HAC_df[HAC_df['file_year'].isin(['2015', '2016', '2017', '2018', '2019', '2020', '2021'])]
hac_months = sorted(list(set(HAC_df['file_month'].tolist())))
hac_years = sorted(list(set(HAC_df['file_year'].tolist())))

for yr in hac_years:
    df1 = HAC_df[HAC_df['file_year'] == yr]
    for mo in hac_months:
        hac_df = df1[df1['file_month'] == mo]
        if hac_df.shape[0] == 0:
            continue
        else:
            hac_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
            hac_df.dropna(how='all', axis=1, inplace=True)
            hac_df = hac_df[~hac_df['Total HAC Score'].isin([np.nan, float('NaN')])]
            hac_df.sort_values(by='Facility ID', inplace=True)
            hac_hosps = sorted(list(set(hac_df['Facility ID'].tolist())))
            
            try:
                start_date = sorted(list(set(hac_df['HAI Measures Start Date'].tolist())))
                start_date = start_date[0]
                
                end_date = sorted(list(set(hac_df['HAI Measures End Date'].tolist())))
                end_date = end_date[0]
                

            except:
                start_date = sorted(list(set(hac_df['Domain 2 Start Date'].tolist())))
                start_date = start_date[0]
                
                end_date = sorted(list(set(hac_df['Domain 2 End Date'].tolist())))
                end_date = end_date[0]
            
            #########################################################################################
            ##############################   MRSA   #################################################
            #########################################################################################

            df = pd.read_pickle(mydir + "1_preprocess_CareCompare_data/preprocessed_HAI_data/MRSA_Data.pkl")
            df['MRSA Predicted Cases'] = pd.to_numeric(df['MRSA Predicted Cases'], errors='coerce')
            df['MRSA patient days'] = pd.to_numeric(df['MRSA patient days'], errors='coerce')
            df['MRSA Observed Cases'] = pd.to_numeric(df['MRSA Observed Cases'], errors='coerce')
            
            start_df = df[df['Start Date'] == start_date]
            start_file_dates = start_df['file_month'] + '-' + start_df['file_year']
            start_file_dates = sorted(list(set(start_file_dates)))
            
            end_df = df[df['End Date'] == end_date]
            end_file_dates = end_df['file_month'] + '-' + end_df['file_year']
            end_file_dates = sorted(list(set(end_file_dates)))
            
            for start_file_date in start_file_dates:
                start_file_mo = start_file_date[0:2]
                start_file_yr = start_file_date[-4:]
                start_df2 = start_df[(start_df['file_month'] == start_file_mo) & (start_df['file_year'] == start_file_yr)]
                measure_end_date = list(set(start_df2['End Date'].tolist()))
                
                if len(measure_end_date) > 1:
                    continue
                
                for end_file_date in end_file_dates:
                    end_mo = end_file_date[0:2]
                    end_yr = end_file_date[-4:]
                    end_df2 = end_df[(end_df['file_month'] == end_mo) & (end_df['file_year'] == end_yr)]
                    
                    end_df_start_dates = list(set(end_df2['Start Date'].tolist()))
                    tdf = start_df2.copy(deep=True)
                    tdf = tdf.merge(end_df2, how='outer', on='Facility ID')
                    
                    tdf['MRSA Predicted Cases_x'].fillna(0, inplace=True) 
                    tdf['MRSA Predicted Cases_y'].fillna(0, inplace=True)
                    tdf['MRSA Observed Cases_x'].fillna(0, inplace=True) 
                    tdf['MRSA Observed Cases_y'].fillna(0, inplace=True) 
                    
                    tdf['MRSA Predicted Cases'] = tdf['MRSA Predicted Cases_x'] + tdf['MRSA Predicted Cases_y']
                    tdf['MRSA patient days'] = tdf['MRSA patient days_x'] + tdf['MRSA patient days_y']
                    tdf['MRSA Observed Cases'] = tdf['MRSA Observed Cases_x'] + tdf['MRSA Observed Cases_y']
                    tdf['Start Date'] = tdf['Start Date_x'].tolist()
                    tdf['End Date'] = tdf['End Date_y'].tolist()
                    
                    tdf.drop(labels=['Start Date_x', 'End Date_x', 'MRSA Predicted Cases_x', 
                                     'MRSA patient days_x', 'MRSA_x', 'MRSA upper CL_x', 'MRSA lower CL_x', 
                                     'MRSA Observed Cases_x', 'file_month_x', 'file_year_x', 
                                     'Start Date_y', 'End Date_y', 'MRSA Predicted Cases_y', 
                                     'MRSA patient days_y', 'MRSA_y', 'MRSA upper CL_y', 
                                     'MRSA lower CL_y', 'MRSA Observed Cases_y', 
                                     'file_month_y', 'file_year_y'], axis=1, inplace=True)
                    
                    #tdf = tdf[tdf['MRSA Predicted Cases'] >= 1]
                    tdf = tdf[tdf['Facility ID'].isin(hac_hosps)]
                    tdf['SIR'] = tdf['MRSA Observed Cases']/tdf['MRSA Predicted Cases']
                    tdf['Winzorized SIR'] = Winsorize_it(tdf['SIR'].tolist())
                    tdf['MRSA SIR W Z Score'] = ZScore_it(tdf['Winzorized SIR'])
                    #tdf.dropna(how='any', subset=['MRSA SIR W Z Score'], inplace=True)
                    #hac_df.dropna(how='any', subset=['MRSA W Z Score'], inplace=True)
                    
                    #tdf = tdf[~tdf['MRSA SIR W Z Score'].isin([np.nan, float('NaN')])]
                    #hac_df = hac_df[~hac_df['MRSA W Z Score'].isin([np.nan, float('NaN')])]
                    
                    #print(list(tdf))
                    #print(list(hac_df))
                    #print(tdf.head())
                    if tdf.shape[0] == hac_df.shape[0]:
                        print('HACRP file date:', mo, '/', yr)
                        print('Start date:', start_date)
                        print('End date:', end_date)
                        
                        print(tdf.shape[0])
                        print(hac_df.shape[0])
                        
                        tdf.sort_values(by='Facility ID', inplace=True)
                        hac_df.sort_values(by='Facility ID', inplace=True)
                        
                        r = 1
                        threshold = 0.1 #0.0075
                        misses = 0
                        for i in range(hac_df.shape[0]):
                            v1 = tdf['MRSA SIR W Z Score'].iloc[i]
                            h1 = tdf['Facility ID'].iloc[i]
                            v2 = hac_df['MRSA W Z Score'].iloc[i]
                            h2 = hac_df['Facility ID'].iloc[i]
                            if h1 == h2:
                                if np.isnan(v1) == False and np.isnan(v2) == False:
                                    e = err(v1, v2)
                                    if e > threshold:
                                        misses += 1
                                        #print(i, 'MRSA:', v1, v2)
                            else:
                                print('h1 != h2')
                                sys.exit()
                                
                        print('miss rate:', 100*misses/hac_df.shape[0], '\n')

                #sys.exit()
            '''
            ci_alpha = 0.4
            b_method = 'normal'
            fdate = yr + '_' + mo

            hai_file_dates = hai_df['file_month'] + '-' + hai_df['file_year']
            hai_file_dates = sorted(list(set(hai_file_dates)))
            hai_mos = sorted(list(set(hai_df['file_month'].tolist())))
            hai_yrs = sorted(list(set(hai_df['file_year'].tolist())))
            
            print(hai_file_dates)
            print(list(hai_df))
            
            for dt in hai_file_dates:
                mo = dt[0:2]
                yr = dt[-4:]
                
                mrsa_df = hai_df[hai_df['file_month'] == mo]
                mrsa_df = mrsa_df[mrsa_df['file_year'] == yr]
                
                mrsa_df.drop_duplicates(inplace=True)
                mrsa_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
                mrsa_df = mrsa_df[mrsa_df['Facility ID'].isin(hac_hosps)]
                mrsa_df.rename(columns={
                    'MRSA Predicted Cases': 'Predicted Cases',
                    'MRSA patient days': 'Days',
                    'MRSA': 'SIR',
                    'MRSA Observed Cases': 'Observed Cases',
                    }, inplace=True)
                cols = ['Predicted Cases', 'Days', 'SIR', 'Observed Cases']
                for col in cols: mrsa_df[col] = pd.to_numeric(mrsa_df[col], errors='coerce')
                
                print(len(hac_hosps), 'hospitals in HACRP file')
                print(len(list(set(mrsa_df['Facility ID'].tolist()))), 'hospitals in MRSA file')
                
            
            sys.exit()
            '''

'''            

tdf = pd.read_pickle(mydir + "optimize/optimized_by_quarter/MRSA/MRSA_Data_opt_for_SIRs_" + fdate + ".pkl")
tdf.drop_duplicates(inplace=True)
tdf = tdf[tdf['Facility ID'].isin(hac_hosps)]
days = np.array(tdf['MRSA patient days'])
ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
tdf['exp_random'] = days * ci_upp
tdf['SIS'] = (tdf['MRSA Observed Cases'] - tdf['exp_random']) / tdf['MRSA Predicted Cases']
tdf = tdf.filter(items=['Facility ID', 'SIS'], axis=1)

mrsa_df = mrsa_df.merge(tdf, on='Facility ID', how='outer')
mrsa_df['Winzorized SIR'] = Winsorize_it(mrsa_df['SIR'].tolist())
mrsa_df['MRSA SIR W Z Score'] = ZScore_it(mrsa_df['Winzorized SIR'])
mrsa_df['Winzorized SIS'] = Winsorize_it(mrsa_df['SIS'].tolist())
mrsa_df['MRSA SIS W Z Score'] = ZScore_it(mrsa_df['Winzorized SIS'])
mrsa_df.drop(labels=['Winzorized SIR', 'Winzorized SIS', 
                     'MRSA upper CL', 'MRSA lower CL'], axis=1, inplace=True)

print(len(list(set(mrsa_df['Facility ID'].tolist()))), 'hospitals in MRSA file')
mrsa_df.sort_values(by='Facility ID', inplace=True)
#print('mrsa_df.shape', mrsa_df.shape[0])
#print(mrsa_df.head())
#sys.exit()

#########################################################################################
##############################   CDIFF   ################################################
#########################################################################################

cdiff_df = pd.read_pickle(mydir + "CareCompare_data/preprocessed_HAC/CDIFF_Data.pkl")
cdiff_df = cdiff_df[cdiff_df['file_month'] == mo]
cdiff_df = cdiff_df[cdiff_df['file_year'] == yr]
cdiff_df.drop_duplicates(inplace=True)
cdiff_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
cdiff_df.dropna(how='all', axis=1, inplace=True)
cdiff_df = cdiff_df[cdiff_df['Facility ID'].isin(hac_hosps)]
cdiff_df.rename(columns={
    'CDIFF Predicted Cases': 'Predicted Cases',
    'CDIFF patient days': 'Days',
    'CDIFF': 'SIR',
    'CDIFF Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'Days', 'SIR', 'Observed Cases']
for col in cols: cdiff_df[col] = pd.to_numeric(cdiff_df[col], errors='coerce')
    
tdf = pd.read_pickle(mydir + "optimize/optimized_by_quarter/CDIFF/CDIFF_Data_opt_for_SIRs_" + fdate + ".pkl")
tdf.drop_duplicates(inplace=True)
tdf = tdf[tdf['Facility ID'].isin(hac_hosps)]
days = np.array(tdf['CDIFF patient days'])
ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
tdf['exp_random'] = days * ci_upp
tdf['SIS'] = (tdf['CDIFF Observed Cases'] - tdf['exp_random']) / tdf['CDIFF Predicted Cases']
tdf = tdf.filter(items=['Facility ID', 'SIS'], axis=1)

cdiff_df = cdiff_df.merge(tdf, on='Facility ID', how='outer')
cdiff_df['Winzorized SIR'] = Winsorize_it(cdiff_df['SIR'].tolist())
cdiff_df['CDI SIR W Z Score'] = ZScore_it(cdiff_df['Winzorized SIR'])
cdiff_df['Winzorized SIS'] = Winsorize_it(cdiff_df['SIS'].tolist())
cdiff_df['CDI SIS W Z Score'] = ZScore_it(cdiff_df['Winzorized SIS'])
cdiff_df.drop(labels=['Winzorized SIR', 'Winzorized SIS', 
                     'CDIFF upper CL', 'CDIFF lower CL'], axis=1, inplace=True)

print(len(list(set(cdiff_df['Facility ID'].tolist()))), 'hospitals in CDIFF file')
cdiff_df.sort_values(by='Facility ID', inplace=True)
#print('cdiff_df.shape', cdiff_df.shape[0])
#print(cdiff_df.head())
#sys.exit()


#########################################################################################
##############################   CAUTI   ################################################
#########################################################################################

cauti_df = pd.read_pickle(mydir + "CareCompare_data/preprocessed_HAC/CAUTI_Data.pkl")
cauti_df = cauti_df[cauti_df['file_month'] == mo]
cauti_df = cauti_df[cauti_df['file_year'] == yr]
cauti_df.drop_duplicates(inplace=True)
cauti_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
cauti_df.dropna(how='all', axis=1, inplace=True)
cauti_df = cauti_df[cauti_df['Facility ID'].isin(hac_hosps)]
cauti_df.rename(columns={
    'CAUTI Predicted Cases': 'Predicted Cases',
    'CAUTI Urinary Catheter Days': 'Days',
    'CAUTI': 'SIR',
    'CAUTI Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'Days', 'SIR', 'Observed Cases']
for col in cols: cauti_df[col] = pd.to_numeric(cauti_df[col], errors='coerce')

tdf = pd.read_pickle(mydir + "optimize/optimized_by_quarter/CAUTI/CAUTI_Data_opt_for_SIRs_" + fdate + ".pkl")
tdf.drop_duplicates(inplace=True)
tdf = tdf[tdf['Facility ID'].isin(hac_hosps)]
days = np.array(tdf['CAUTI Urinary Catheter Days'])
ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
tdf['exp_random'] = days * ci_upp
tdf['SIS'] = (tdf['CAUTI Observed Cases'] - tdf['exp_random']) / tdf['CAUTI Predicted Cases']
tdf = tdf.filter(items=['Facility ID', 'SIS'], axis=1)

cauti_df = cauti_df.merge(tdf, on='Facility ID', how='outer')
cauti_df['Winzorized SIR'] = Winsorize_it(cauti_df['SIR'].tolist())
cauti_df['CAUTI SIR W Z Score'] = ZScore_it(cauti_df['Winzorized SIR'])
cauti_df['Winzorized SIS'] = Winsorize_it(cauti_df['SIS'].tolist())
cauti_df['CAUTI SIS W Z Score'] = ZScore_it(cauti_df['Winzorized SIS'])
cauti_df.drop(labels=['Winzorized SIR', 'Winzorized SIS', 
                     'CAUTI upper CL', 'CAUTI lower CL'], axis=1, inplace=True)

print(len(list(set(cauti_df['Facility ID'].tolist()))), 'hospitals in CAUTI file')
cauti_df.sort_values(by='Facility ID', inplace=True)
#print('cauti_df.shape', cauti_df.shape[0])
#print(cauti_df.head())
#sys.exit()

#########################################################################################
##############################   CLABSI   ###############################################
#########################################################################################

clabsi_df = pd.read_pickle(mydir + "CareCompare_data/preprocessed_HAC/CLABSI_Data.pkl")
clabsi_df = clabsi_df[clabsi_df['file_month'] == mo]
clabsi_df = clabsi_df[clabsi_df['file_year'] == yr]
clabsi_df.drop_duplicates(inplace=True)
clabsi_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
clabsi_df.dropna(how='all', axis=1, inplace=True)
clabsi_df = clabsi_df[clabsi_df['Facility ID'].isin(hac_hosps)]
clabsi_df.rename(columns={
    'CLABSI Predicted Cases': 'Predicted Cases',
    'CLABSI Number of Device Days': 'Days',
    'CLABSI': 'SIR',
    'CLABSI Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'Days', 'SIR', 'Observed Cases']
for col in cols: clabsi_df[col] = pd.to_numeric(clabsi_df[col], errors='coerce')
    
tdf = pd.read_pickle(mydir + "optimize/optimized_by_quarter/CLABSI/CLABSI_Data_opt_for_SIRs_" + fdate + ".pkl")
tdf.drop_duplicates(inplace=True)
tdf = tdf[tdf['Facility ID'].isin(hac_hosps)]
days = np.array(tdf['CLABSI Number of Device Days'])
ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
tdf['exp_random'] = days * ci_upp
tdf['SIS'] = (tdf['CLABSI Observed Cases'] - tdf['exp_random']) / tdf['CLABSI Predicted Cases']
tdf = tdf.filter(items=['Facility ID', 'SIS'], axis=1)

clabsi_df = clabsi_df.merge(tdf, on='Facility ID', how='outer')
clabsi_df['Winzorized SIR'] = Winsorize_it(clabsi_df['SIR'].tolist())
clabsi_df['CLABSI SIR W Z Score'] = ZScore_it(clabsi_df['Winzorized SIR'])
clabsi_df['Winzorized SIS'] = Winsorize_it(clabsi_df['SIS'].tolist())
clabsi_df['CLABSI SIS W Z Score'] = ZScore_it(clabsi_df['Winzorized SIS'])
clabsi_df.drop(labels=['Winzorized SIR', 'Winzorized SIS', 
                     'CLABSI upper CL', 'CLABSI lower CL'], axis=1, inplace=True)

print(len(list(set(clabsi_df['Facility ID'].tolist()))), 'hospitals in CLABSI file')
clabsi_df.sort_values(by='Facility ID', inplace=True)
#print('clabsi_df.shape', clabsi_df.shape[0])
#print(clabsi_df.head())
#sys.exit()


#########################################################################################
##############  Check lists of hospitals for agreement in dataframes  ###################
#########################################################################################
print('Check lists of hospitals for agreement in dataframes:')

p1 = hac_df['Facility ID'].tolist()
p2 = mrsa_df['Facility ID'].tolist()
p3 = cdiff_df['Facility ID'].tolist()
p4 = cauti_df['Facility ID'].tolist()
p5 = clabsi_df['Facility ID'].tolist()

ls = [len(p1), len(p2), len(p3), len(p4), len(p5)]
if len(list(set(ls))) > 1:
    print(ls)
    sys.exit()

for i, h in enumerate(p1):
    ls = [h, p2[i], p3[i], p4[i], p5[i]] 
    if len(list(set(ls))) == 1:
        pass
    else:
        print('Hospital lists differ:')
        print(ls)
        sys.exit()
        
print('Hospitals lists are good\n')


#########################################################################################
############### New HAC dataframe with replacement scores ###############################
#################### Check agreement of scores ##########################################

print('Check agreement of scores:')

hac_df2 = hac_df.copy(deep=True)
hac_df2['CAUTI W Z Score (derived)'] = cauti_df['CAUTI SIR W Z Score'].tolist()
hac_df2['CLABSI W Z Score (derived)'] = clabsi_df['CLABSI SIR W Z Score'].tolist()
hac_df2['MRSA W Z Score (derived)'] = mrsa_df['MRSA SIR W Z Score'].tolist()
hac_df2['CDI W Z Score (derived)'] = cdiff_df['CDI SIR W Z Score'].tolist()

hac_df2['CAUTI SIS W Z Score'] = cauti_df['CAUTI SIS W Z Score'].tolist()
hac_df2['CLABSI SIS W Z Score'] = clabsi_df['CLABSI SIS W Z Score'].tolist()
hac_df2['MRSA SIS W Z Score'] = mrsa_df['MRSA SIS W Z Score'].tolist()
hac_df2['CDI SIS W Z Score'] = cdiff_df['CDI SIS W Z Score'].tolist()

cols = ['Predicted Cases', 'Days', 'SIR', 'SIS', 'Observed Cases']
for c in cols:
    hac_df2['CAUTI ' + c] = cauti_df[c].tolist() 
    hac_df2['CLABSI ' + c] = clabsi_df[c].tolist() 
    hac_df2['CDI ' + c] = cdiff_df[c].tolist() 
    hac_df2['MRSA ' + c] = mrsa_df[c].tolist() 


#########################################################################################
############################ Get HAC Scores #############################################
#########################################################################################

######################## Based on derived SIR ###########################################

hac_scores = []
for hosp in hac_df2['Facility ID'].tolist():
    tdf = hac_df2[hac_df2['Facility ID'] == hosp]
    
    w_ls = []
    sum_ls = []
    m_ls = ['CDI W Z Score (derived)', 'CAUTI W Z Score (derived)', 
            'CLABSI W Z Score (derived)', 'MRSA W Z Score (derived)', 
            'SSI W Z Score', 'PSI-90 W Z Score']
    
    s = 0
    w = np.nan
    for m in m_ls:
        v = tdf[m].iloc[0]
        
        if np.isnan(v) == False: 
            s += v
            w_ls.append(v)
            
    if len(w_ls) > 0:
        w = 1 / len(w_ls)
        hac_scores.append(s*w)
    else:
        w = np.nan
        hac_scores.append(np.nan)
        
hac_df2['Total HAC Score (derived)'] = hac_scores

tdf = hac_df2[~hac_df2['Total HAC Score (derived)'].isin([np.nan, float('NaN')])]
p75 = np.percentile(tdf['Total HAC Score (derived)'], 75)
print('p75:', p75)

pr = []
for hosp in hac_df2['Facility ID'].tolist():
    tdf = hac_df[hac_df['Facility ID'] == hosp]
    
    p = tdf['Payment Reduction'].iloc[0]
    if p != 'Yes' and p != 'No' and np.isnan(p) == True:
        pr.append(np.nan)
    
    else:
        tdf = hac_df2[hac_df2['Facility ID'] == hosp]
        score = tdf['Total HAC Score (derived)'].iloc[0]
        
        if np.isnan(score) == True:
            pr.append('No')
        elif score <= p75:
            pr.append('No')
        elif score > p75:
            pr.append('Yes')
        else:
            print(score)
            sys.exit()


hac_df2['Payment Reduction (derived)'] = pr

########################## Based on SIS #################################################

hac_scores = []
for hosp in hac_df2['Facility ID'].tolist():
    tdf = hac_df2[hac_df2['Facility ID'] == hosp]
    
    w_ls = []
    sum_ls = []
    m_ls = ['CDI SIS W Z Score', 'CAUTI SIS W Z Score', 
            'CLABSI SIS W Z Score', 'MRSA SIS W Z Score',
            'SSI W Z Score', 'PSI-90 W Z Score']
    
    s = 0
    w = np.nan
    for m in m_ls:
        v = tdf[m].iloc[0]
        
        if np.isnan(v) == False: 
            s += v
            w_ls.append(v)
            
    if len(w_ls) > 0:
        w = 1 / len(w_ls)
        hac_scores.append(s*w)
    else:
        w = np.nan
        hac_scores.append(np.nan)
        
hac_df2['Total HAC Score (SIS)'] = hac_scores

tdf = hac_df2[~hac_df2['Total HAC Score (SIS)'].isin([np.nan, float('NaN')])]
p75 = np.percentile(tdf['Total HAC Score (SIS)'], 75)
print('p75:', p75)

pr = []
for hosp in hac_df2['Facility ID'].tolist():
    tdf = hac_df[hac_df['Facility ID'] == hosp]
    
    p = tdf['Payment Reduction'].iloc[0]
    if p != 'Yes' and p != 'No' and np.isnan(p) == True:
        pr.append(np.nan)
    
    else:
        tdf = hac_df2[hac_df2['Facility ID'] == hosp]
        score = tdf['Total HAC Score (SIS)'].iloc[0]
        
        if np.isnan(score) == True:
            pr.append('No')
        elif score <= p75:
            pr.append('No')
        elif score > p75:
            pr.append('Yes')
        else:
            print(score)
            sys.exit()


hac_df2['Payment Reduction (SIS)'] = pr



#########################################################################################
########################### Check agreement #############################################
#########################################################################################

r = 1
threshold = 0.0075
misses = 0
for i in range(hac_df.shape[0]):
    
    v1 = hac_df['CAUTI W Z Score'].iloc[i]
    v2 = hac_df2['CAUTI W Z Score (derived)'].iloc[i]
    if np.isnan(v1) == False and np.isnan(v2) == False:
        e = err(v1, v2)
        if e > threshold:
            print(i, 'CAUTI:', v1, v2, '\n')
            sys.exit()
        
    v1 = hac_df['CLABSI W Z Score'].iloc[i]
    v2 = hac_df2['CLABSI W Z Score (derived)'].iloc[i]
    if np.isnan(v1) == False and np.isnan(v2) == False:
        e = err(v1, v2)
        if e > threshold:
            print(i, 'CLABSI:', v1, v2)
            sys.exit()
        
    v1 = hac_df['CDI W Z Score'].iloc[i]
    v2 = hac_df2['CDI W Z Score (derived)'].iloc[i]
    if np.isnan(v1) == False and np.isnan(v2) == False:
        e = err(v1, v2)
        if e > threshold:
            print(i, 'CDI:', v1, v2)
            sys.exit()
        
    v1 = hac_df['MRSA W Z Score'].iloc[i]
    v2 = hac_df2['MRSA W Z Score (derived)'].iloc[i]
    if np.isnan(v1) == False and np.isnan(v2) == False:
        e = err(v1, v2)
        if e > threshold:
            print(i, 'MRSA:', v1, v2)
            sys.exit()
    
    if 1 == 1:
        v1 = hac_df['Total HAC Score'].iloc[i]
        v2 = hac_df2['Total HAC Score (derived)'].iloc[i]
        if np.isnan(v1) == False and np.isnan(v2) == False:
            e = err(v1, v2)
            if e > threshold:
                print(i, 'Total HAC Score:', v1, v2)
                #sys.exit()
    
    if 1 == 1:
        v1 = hac_df['Payment Reduction'].iloc[i]
        v2 = hac_df2['Payment Reduction (derived)'].iloc[i]
        if v1 in ['Yes', 'No'] and v2 in ['Yes', 'No'] and v1 != v2:
            misses += 1
            #print(i, 'Payment Reduction:', v1, v2)
            #sys.exit()    
    
print('Misses:', misses)
print(hac_df2.shape)
print(list(hac_df2))

#hac_df2.to_pickle(mydir + 'yearly_compiled/HACRP-File-' + hac_mo + '-' + hac_yr + '_HAI-File-' + mo + '-' + yr + '.pkl')
'''













