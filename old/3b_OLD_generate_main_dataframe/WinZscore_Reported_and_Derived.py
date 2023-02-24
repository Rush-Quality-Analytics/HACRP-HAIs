import pandas as pd
import numpy as np
import random
import sys
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
# ######################   CUSTOM FXNS   #################################################
# ########################################################################################

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
    zscores = stats.zscore(x, nan_policy='omit')
    return zscores


def err(T, E):
    e = np.abs([E - T])
    return e


def SIS(O, E, P, uP, uE, V):
    
    # Doesn't work:
     # sis = (O - E) * P
     # sis = (O - uE) * P
     # sis = (O - E) * (uP - (O/P))
     # sis = ((O - uE)/uE) + (uP - (O/P)/uP)
     # sis = (O - E) * (uP - O/P)
     # sis = (O - uE) * (O/P)
     # sis = (O - uE) * uP
     # sis = (O - uE) / uP
     # sis = ((O - P) + (O - uE) + (P - uE))/np.sqrt(V)
     
    # Works: 
    sis = (O - E) / P
    
    
    return sis
    

#########################################################################################
# #######################   HACRP DATA   #################################################
# ########################################################################################

# month and year for HACRP data
hac_mo = '04'
hac_yr = '2022'

hac_df = pd.read_pickle(mydir + "data/CareCompare_data/CombinedFiles_HACRP/Facility.pkl")
hac_df = hac_df[hac_df['file_month'] == hac_mo]
hac_df = hac_df[hac_df['file_year'] == hac_yr]
hac_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
hac_df.dropna(how='all', axis=1, inplace=True)
hac_df['Total HAC Score'] = pd.to_numeric(hac_df['Total HAC Score'], errors='coerce')
hac_df = hac_df[~hac_df['Total HAC Score'].isin([np.nan, float('NaN'), 'N/A'])]
hac_df.sort_values(by='Facility ID', inplace=True)
hac_hosps = list(set(hac_df['Facility ID'].tolist()))
print(len(hac_hosps), 'hospitals in HACRP file')

start_date = list(set(hac_df['HAI Measures Start Date'].tolist()))
start_date = start_date[0]
#print('HACRP: HAI Measures Start Date:', start_date)

end_date = list(set(hac_df['HAI Measures End Date'].tolist()))
end_date = end_date[0]
#print('HACRP: HAI Measures End Date:', end_date, '\n')

# month and year for corresponding HAI data
mo = '04'
yr = '2021'

ci_alpha = 0.5
b_method = 'normal'
fdate = yr + '_' + mo

sis_var = 'expected O'
#sis_var = 'exp_random'

#########################################################################################
# #############################   MRSA   #################################################
# ########################################################################################

mrsa_df = pd.read_pickle(mydir + "1_preprocess_CareCompare_data/preprocessed_HAI_data/MRSA_Data.pkl")
mrsa_df = mrsa_df[mrsa_df['Start Date'] == start_date]
mrsa_df = mrsa_df[mrsa_df['End Date'] == end_date]
mrsa_df = mrsa_df[mrsa_df['file_month'] == mo]
mrsa_df = mrsa_df[mrsa_df['file_year'] == yr]
mrsa_df.drop_duplicates(inplace=True)
mrsa_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
mrsa_df.dropna(how='all', axis=1, inplace=True)

mrsa_df = mrsa_df[mrsa_df['Facility ID'].isin(hac_hosps)]
mrsa_df.rename(columns={
    'MRSA Predicted Cases': 'Predicted Cases',
    'MRSA patient days': 'Volume',
    'MRSA': 'SIR',
    'MRSA upper CL': 'upper CL',
    'MRSA Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'Volume', 'SIR', 'Observed Cases', 'upper CL']
for col in cols: mrsa_df[col] = pd.to_numeric(mrsa_df[col], errors='coerce')

start_date_mrsa = list(set(mrsa_df['Start Date'].tolist()))
end_date_mrsa = list(set(mrsa_df['End Date'].tolist()))

tdf = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/MRSA/MRSA_Data_opt_for_SIRs_" + fdate + ".pkl")
tdf.drop_duplicates(inplace=True)
print(tdf.shape[0], 'hospitals in MRSA file')

tdf.rename(columns={
    'MRSA Predicted Cases': 'Predicted Cases',
    'MRSA': 'SIR',
    'MRSA upper CL': 'upper CL',
    'MRSA Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'SIR', 'Observed Cases', 'upper CL']
for col in cols: tdf[col] = pd.to_numeric(tdf[col], errors='coerce')

tdf = tdf[tdf['Facility ID'].isin(hac_hosps)]
days = np.array(tdf['MRSA patient days'])
ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
tdf['exp_random'] = days * ci_upp
tdf['SIS'] = SIS(tdf['Observed Cases'], 
                 tdf['expected O'], 
                 tdf['Predicted Cases'], 
                 tdf['upper CL'], 
                 tdf['exp_random'],
                 days)

tdf['SIR (random)'] = tdf[sis_var] / tdf['Predicted Cases']
tdf = tdf.filter(items=['Facility ID', 'SIS', 'SIR (random)'], axis=1)

mrsa_df = mrsa_df.merge(tdf, on='Facility ID', how='outer')
mrsa_df['Winsorized SIR'] = Winsorize_it(mrsa_df['SIR'].tolist())
mrsa_df['MRSA SIR W Z Score'] = ZScore_it(mrsa_df['Winsorized SIR'])
mrsa_df['Winsorized SIS'] = Winsorize_it(mrsa_df['SIS'].tolist())
mrsa_df['MRSA SIS W Z Score'] = ZScore_it(mrsa_df['Winsorized SIS'])
mrsa_df['Winsorized SIR (random)'] = Winsorize_it(mrsa_df['SIR (random)'].tolist())
mrsa_df['MRSA W Z Score (random)'] = ZScore_it(mrsa_df['Winsorized SIR (random)'])
mrsa_df.drop(labels=['Winsorized SIR', 'Winsorized SIS', 'Winsorized SIR (random)', 'upper CL', 'MRSA lower CL'], axis=1, inplace=True)
mrsa_df.sort_values(by='Facility ID', inplace=True)

#########################################################################################
# #############################   CDIFF   ################################################
# ########################################################################################

cdiff_df = pd.read_pickle(mydir + "1_preprocess_CareCompare_data/preprocessed_HAI_data/CDIFF_Data.pkl")
cdiff_df = cdiff_df[cdiff_df['Start Date'] == start_date]
cdiff_df = cdiff_df[cdiff_df['End Date'] == end_date]
cdiff_df = cdiff_df[cdiff_df['file_month'] == mo]
cdiff_df = cdiff_df[cdiff_df['file_year'] == yr]
cdiff_df.drop_duplicates(inplace=True)
cdiff_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
cdiff_df.dropna(how='all', axis=1, inplace=True)

cdiff_df = cdiff_df[cdiff_df['Facility ID'].isin(hac_hosps)]
cdiff_df.rename(columns={
    'CDIFF Predicted Cases': 'Predicted Cases',
    'CDIFF patient days': 'Volume',
    'CDIFF': 'SIR',
    'CDIFF upper CL': 'upper CL',
    'CDIFF Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'Volume', 'SIR', 'Observed Cases', 'upper CL']
for col in cols: cdiff_df[col] = pd.to_numeric(cdiff_df[col], errors='coerce')

start_date_cdiff = list(set(cdiff_df['Start Date'].tolist()))
end_date_cdiff = list(set(cdiff_df['End Date'].tolist()))

tdf = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/CDIFF/CDIFF_Data_opt_for_SIRs_" + fdate + ".pkl")
tdf.drop_duplicates(inplace=True)
print(tdf.shape[0], 'hospitals in CDIFF file')

tdf.rename(columns={
    'CDIFF Predicted Cases': 'Predicted Cases',
    'CDIFF': 'SIR',
    'CDIFF upper CL': 'upper CL',
    'CDIFF Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'SIR', 'Observed Cases', 'upper CL']
for col in cols: tdf[col] = pd.to_numeric(tdf[col], errors='coerce')


tdf = tdf[tdf['Facility ID'].isin(hac_hosps)]
days = np.array(tdf['CDIFF patient days'])
ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
tdf['exp_random'] = days * ci_upp
tdf['SIS'] = SIS(tdf['Observed Cases'], 
                 tdf['expected O'], 
                 tdf['Predicted Cases'], 
                 tdf['upper CL'], 
                 tdf['exp_random'],
                 days)

tdf['SIR (random)'] = tdf[sis_var] / tdf['Predicted Cases']
tdf = tdf.filter(items=['Facility ID', 'SIS', 'SIR (random)'], axis=1)

cdiff_df = cdiff_df.merge(tdf, on='Facility ID', how='outer')
cdiff_df['Winsorized SIR'] = Winsorize_it(cdiff_df['SIR'].tolist())
cdiff_df['CDI SIR W Z Score'] = ZScore_it(cdiff_df['Winsorized SIR'])
cdiff_df['Winsorized SIS'] = Winsorize_it(cdiff_df['SIS'].tolist())
cdiff_df['CDI SIS W Z Score'] = ZScore_it(cdiff_df['Winsorized SIS'])
cdiff_df['Winsorized SIR (random)'] = Winsorize_it(cdiff_df['SIR (random)'].tolist())
cdiff_df['CDI W Z Score (random)'] = ZScore_it(cdiff_df['Winsorized SIR (random)'])
cdiff_df.drop(labels=['Winsorized SIR', 'Winsorized SIS', 'Winsorized SIR (random)', 'upper CL', 'CDIFF lower CL'], axis=1, inplace=True)
cdiff_df.sort_values(by='Facility ID', inplace=True)


#########################################################################################
# #############################   CAUTI   ################################################
# ########################################################################################

cauti_df = pd.read_pickle(mydir + "1_preprocess_CareCompare_data/preprocessed_HAI_data/CAUTI_Data.pkl")
cauti_df = cauti_df[cauti_df['Start Date'] == start_date]
cauti_df = cauti_df[cauti_df['End Date'] == end_date]
cauti_df = cauti_df[cauti_df['file_month'] == mo]
cauti_df = cauti_df[cauti_df['file_year'] == yr]
cauti_df.drop_duplicates(inplace=True)
cauti_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
cauti_df.dropna(how='all', axis=1, inplace=True)

cauti_df = cauti_df[cauti_df['Facility ID'].isin(hac_hosps)]
cauti_df.rename(columns={
    'CAUTI Predicted Cases': 'Predicted Cases',
    'CAUTI Urinary Catheter Days': 'Volume',
    'CAUTI': 'SIR',
    'CAUTI upper CL': 'upper CL',
    'CAUTI Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'Volume', 'SIR', 'Observed Cases', 'upper CL']
for col in cols: cauti_df[col] = pd.to_numeric(cauti_df[col], errors='coerce')

start_date_cauti = list(set(cauti_df['Start Date'].tolist()))
end_date_cauti = list(set(cauti_df['End Date'].tolist()))

tdf = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/CAUTI/CAUTI_Data_opt_for_SIRs_" + fdate + ".pkl")
tdf.drop_duplicates(inplace=True)
print(tdf.shape[0], 'hospitals in CAUTI file')

tdf.rename(columns={
    'CAUTI Predicted Cases': 'Predicted Cases',
    'CAUTI': 'SIR',
    'CAUTI upper CL': 'upper CL',
    'CAUTI Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'SIR', 'Observed Cases', 'upper CL']
for col in cols: tdf[col] = pd.to_numeric(tdf[col], errors='coerce')


tdf = tdf[tdf['Facility ID'].isin(hac_hosps)]
days = np.array(tdf['CAUTI Urinary Catheter Days'])
ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
tdf['exp_random'] = days * ci_upp
tdf['SIS'] = SIS(tdf['Observed Cases'], 
                 tdf['expected O'], 
                 tdf['Predicted Cases'], 
                 tdf['upper CL'], 
                 tdf['exp_random'],
                 days)
tdf['SIR (random)'] = tdf[sis_var] / tdf['Predicted Cases']
tdf = tdf.filter(items=['Facility ID', 'SIS', 'SIR (random)'], axis=1)

cauti_df = cauti_df.merge(tdf, on='Facility ID', how='outer')
cauti_df['Winsorized SIR'] = Winsorize_it(cauti_df['SIR'].tolist())
cauti_df['CAUTI SIR W Z Score'] = ZScore_it(cauti_df['Winsorized SIR'])
cauti_df['Winsorized SIS'] = Winsorize_it(cauti_df['SIS'].tolist())
cauti_df['CAUTI SIS W Z Score'] = ZScore_it(cauti_df['Winsorized SIS'])
cauti_df['Winsorized SIR (random)'] = Winsorize_it(cauti_df['SIR (random)'].tolist())
cauti_df['CAUTI W Z Score (random)'] = ZScore_it(cauti_df['Winsorized SIR (random)'])
cauti_df.drop(labels=['Winsorized SIR', 'Winsorized SIS', 'Winsorized SIR (random)', 'upper CL', 'CAUTI lower CL'], axis=1, inplace=True)
cauti_df.sort_values(by='Facility ID', inplace=True)


#########################################################################################
# #############################   CLABSI   ###############################################
# ########################################################################################

clabsi_df = pd.read_pickle(mydir + "1_preprocess_CareCompare_data/preprocessed_HAI_data/CLABSI_Data.pkl")
clabsi_df = clabsi_df[clabsi_df['Start Date'] == start_date]
clabsi_df = clabsi_df[clabsi_df['End Date'] == end_date]
clabsi_df = clabsi_df[clabsi_df['file_month'] == mo]
clabsi_df = clabsi_df[clabsi_df['file_year'] == yr]
clabsi_df.drop_duplicates(inplace=True)
clabsi_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
clabsi_df.dropna(how='all', axis=1, inplace=True)

clabsi_df = clabsi_df[clabsi_df['Facility ID'].isin(hac_hosps)]
clabsi_df.rename(columns={
    'CLABSI Predicted Cases': 'Predicted Cases',
    'CLABSI Number of Device Days': 'Volume',
    'CLABSI': 'SIR',
    'CLABSI upper CL': 'upper CL',
    'CLABSI Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'Volume', 'SIR', 'Observed Cases', 'upper CL']
for col in cols: clabsi_df[col] = pd.to_numeric(clabsi_df[col], errors='coerce')

start_date_clabsi = list(set(clabsi_df['Start Date'].tolist()))
end_date_clabsi = list(set(clabsi_df['End Date'].tolist()))

tdf = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/CLABSI/CLABSI_Data_opt_for_SIRs_" + fdate + ".pkl")
tdf.drop_duplicates(inplace=True)
print(tdf.shape[0], 'hospitals in CLABSI file')

tdf.rename(columns={
    'CLABSI Predicted Cases': 'Predicted Cases',
    'CLABSI': 'SIR',
    'CLABSI upper CL': 'upper CL',
    'CLABSI Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'SIR', 'Observed Cases', 'upper CL']
for col in cols: tdf[col] = pd.to_numeric(tdf[col], errors='coerce')


tdf = tdf[tdf['Facility ID'].isin(hac_hosps)]
days = np.array(tdf['CLABSI Number of Device Days'])

ci_low, ci_upp = proportion_confint(tdf['expected O'], days, alpha=ci_alpha, method=b_method)
tdf['exp_random'] = days * ci_upp
tdf['SIS'] = SIS(tdf['Observed Cases'], 
                 tdf['expected O'], 
                 tdf['Predicted Cases'], 
                 tdf['upper CL'], 
                 tdf['exp_random'],
                 days)
tdf['SIR (random)'] = tdf[sis_var] / tdf['Predicted Cases']
tdf = tdf.filter(items=['Facility ID', 'SIS', 'SIR (random)'], axis=1)

clabsi_df = clabsi_df.merge(tdf, on='Facility ID', how='outer')
clabsi_df['Winsorized SIR'] = Winsorize_it(clabsi_df['SIR'].tolist())
clabsi_df['CLABSI SIR W Z Score'] = ZScore_it(clabsi_df['Winsorized SIR'])
clabsi_df['Winsorized SIS'] = Winsorize_it(clabsi_df['SIS'].tolist())
clabsi_df['CLABSI SIS W Z Score'] = ZScore_it(clabsi_df['Winsorized SIS'])
clabsi_df['Winsorized SIR (random)'] = Winsorize_it(clabsi_df['SIR (random)'].tolist())
clabsi_df['CLABSI W Z Score (random)'] = ZScore_it(clabsi_df['Winsorized SIR (random)'])
clabsi_df.drop(labels=['Winsorized SIR', 'Winsorized SIS', 'Winsorized SIR (random)', 'upper CL', 'CLABSI lower CL'], axis=1, inplace=True)
clabsi_df.sort_values(by='Facility ID', inplace=True)


#########################################################################################
# #############################  SSIs  ###################################################
# ########################################################################################

SSI_df = pd.read_pickle(mydir + "1_preprocess_CareCompare_data/preprocessed_HAI_data/SSI_Data.pkl")
SSI_df = SSI_df[SSI_df['Start Date'] == start_date]
SSI_df = SSI_df[SSI_df['End Date'] == end_date]
SSI_df = SSI_df[SSI_df['file_month'] == mo]
SSI_df = SSI_df[SSI_df['file_year'] == yr]
SSI_df.drop_duplicates(inplace=True)
SSI_df.drop(labels=['file_month', 'file_year'], axis=1, inplace=True)
SSI_df.dropna(how='all', axis=1, inplace=True)

SSI_df['SSI upper CL'] = SSI_df['SSI Predicted Cases'].tolist()
SSI_df = SSI_df[SSI_df['Facility ID'].isin(hac_hosps)]
SSI_df.rename(columns={
    'SSI Predicted Cases': 'Predicted Cases',
    'SSI Number of Procedures': 'Volume',
    'SSI': 'SIR',
    'SSI upper CL': 'upper CL',
    'SSI Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'Volume', 'SIR', 'Observed Cases', 'upper CL']
for col in cols: SSI_df[col] = pd.to_numeric(SSI_df[col], errors='coerce')

start_date_SSI = list(set(SSI_df['Start Date'].tolist()))
end_date_SSI = list(set(SSI_df['End Date'].tolist()))

tdf = pd.read_pickle(mydir + "2_optimize_random_sampling_models/optimized_by_HAI_file_date/SSI/SSI_Data_opt_for_SIRs_" + fdate + ".pkl")
tdf.drop_duplicates(inplace=True)
print(tdf.shape[0], 'hospitals in SSI file')

tdf['SSI upper CL'] = tdf['SSI Predicted Cases'].tolist()
tdf.rename(columns={
    'SSI Predicted Cases': 'Predicted Cases',
    'SSI': 'SIR',
    'SSI upper CL': 'upper CL',
    'SSI Observed Cases': 'Observed Cases',
    }, inplace=True)
cols = ['Predicted Cases', 'SIR', 'Observed Cases', 'upper CL']
for col in cols: tdf[col] = pd.to_numeric(tdf[col], errors='coerce')


tdf = tdf[tdf['Facility ID'].isin(hac_hosps)]
volume = np.array(tdf['SSI Number of Procedures'])

ci_low, ci_upp = proportion_confint(tdf['expected O'], volume, alpha=ci_alpha, method=b_method)
tdf['exp_random'] = volume * ci_upp
tdf['SIS'] = SIS(tdf['Observed Cases'], 
                 tdf['expected O'], 
                 tdf['Predicted Cases'], 
                 tdf['upper CL'], 
                 tdf['exp_random'],
                 volume)
tdf['SIR (random)'] = tdf[sis_var] / tdf['Predicted Cases']
tdf = tdf.filter(items=['Facility ID', 'SIS', 'SIR (random)'], axis=1)

SSI_df = SSI_df.merge(tdf, on='Facility ID', how='outer')
SSI_df['Winsorized SIR'] = Winsorize_it(SSI_df['SIR'].tolist())
SSI_df['SSI SIR W Z Score'] = ZScore_it(SSI_df['Winsorized SIR'])
SSI_df['Winsorized SIS'] = Winsorize_it(SSI_df['SIS'].tolist())
SSI_df['SSI SIS W Z Score'] = ZScore_it(SSI_df['Winsorized SIS'])
SSI_df['Winsorized SIR (random)'] = Winsorize_it(SSI_df['SIR (random)'].tolist())
SSI_df['SSI W Z Score (random)'] = ZScore_it(SSI_df['Winsorized SIR (random)'])
SSI_df.drop(labels=['Winsorized SIR', 'Winsorized SIS', 'Winsorized SIR (random)', 'upper CL'], axis=1, inplace=True)
SSI_df.sort_values(by='Facility ID', inplace=True)


#########################################################################################
# #############  Check lists of hospitals for agreement in dataframes  ###################
# ########################################################################################
print('Check lists of hospitals for agreement in dataframes:')

p1 = hac_df['Facility ID'].tolist()
p2 = mrsa_df['Facility ID'].tolist()
p3 = cdiff_df['Facility ID'].tolist()
p4 = cauti_df['Facility ID'].tolist()
p5 = clabsi_df['Facility ID'].tolist()
p6 = SSI_df['Facility ID'].tolist()

ls = [len(p1), len(p2), len(p3), len(p4), len(p5), len(p6)]
if len(list(set(ls))) > 1:
    print('Numbers of facilities across files differs.')
    sys.exit()

for i, h in enumerate(p1):
    ls = [h, p2[i], p3[i], p4[i], p5[i], p6[i]] 
    if len(list(set(ls))) == 1:
        pass
    else:
        print('Order of hospital lists differ among the HAC file and the HAI files:')
        print(ls)
        sys.exit()
        
print('Hospitals lists are good\n')


#########################################################################################
# ############## Add derived WinZ scores to HAC dataframe ################################

hac_df['CAUTI W Z Score (derived)'] = cauti_df['CAUTI SIR W Z Score'].tolist()
hac_df['CLABSI W Z Score (derived)'] = clabsi_df['CLABSI SIR W Z Score'].tolist()
hac_df['MRSA W Z Score (derived)'] = mrsa_df['MRSA SIR W Z Score'].tolist()
hac_df['CDI W Z Score (derived)'] = cdiff_df['CDI SIR W Z Score'].tolist()
hac_df['SSI W Z Score (derived)'] = SSI_df['SSI SIR W Z Score'].tolist()

hac_df['CAUTI W Z Score (SIS)'] = cauti_df['CAUTI SIS W Z Score'].tolist()
hac_df['CLABSI W Z Score (SIS)'] = clabsi_df['CLABSI SIS W Z Score'].tolist()
hac_df['MRSA W Z Score (SIS)'] = mrsa_df['MRSA SIS W Z Score'].tolist()
hac_df['CDI W Z Score (SIS)'] = cdiff_df['CDI SIS W Z Score'].tolist()
hac_df['SSI W Z Score (SIS)'] = SSI_df['SSI SIS W Z Score'].tolist()

hac_df['CAUTI W Z Score (random)'] = cauti_df['CAUTI W Z Score (random)'].tolist()
hac_df['CLABSI W Z Score (random)'] = clabsi_df['CLABSI W Z Score (random)'].tolist()
hac_df['MRSA W Z Score (random)'] = mrsa_df['MRSA W Z Score (random)'].tolist()
hac_df['CDI W Z Score (random)'] = cdiff_df['CDI W Z Score (random)'].tolist()
hac_df['SSI W Z Score (random)'] = SSI_df['SSI W Z Score (random)'].tolist()

cols = ['Predicted Cases', 'Volume', 'SIR', 'SIS', 'Observed Cases', 'SIR (random)']
for c in cols:
    hac_df['CAUTI ' + c] = cauti_df[c].tolist() 
    hac_df['CLABSI ' + c] = clabsi_df[c].tolist() 
    hac_df['CDI ' + c] = cdiff_df[c].tolist() 
    hac_df['MRSA ' + c] = mrsa_df[c].tolist() 
    hac_df['SSI ' + c] = SSI_df[c].tolist()

print(hac_df.shape[0], 'hospitals in hac_df')
#########################################################################################
# ############ Assign maximum WinZ scores to hospitals with HAI footnote 18 ##############
# ########################################################################################

hais = ['CLABSI', 'CAUTI', 'SSI', 'MRSA', 'CDI', 'SSI']
for hai in hais:
    r_df = hac_df[hac_df[hai + ' Footnote'].isin([18, '18', '18 ', ' 18'])]
    for i in list(r_df.index.values):
        hac_df.loc[i, hai + ' W Z Score (derived)'] = np.max(hac_df[hai + ' W Z Score (derived)'])
        hac_df.loc[i, hai + ' W Z Score (SIS)'] = np.max(hac_df[hai + ' W Z Score (SIS)'])
        hac_df.loc[i, hai + ' W Z Score (random)'] = np.max(hac_df[hai + ' W Z Score (random)'])

print(hac_df.shape[0], 'hospitals in hac_df')

#########################################################################################
# ############ Remove WinZ scores calculated from HAI data that ##########################
# ############ have no corresponding value in the HACRP data #############################
# ########################################################################################


prvdr_ls = hac_df['Facility ID'].tolist()

for hai in hais:
    ls1 = hac_df[hai + ' W Z Score'].tolist()
    ls2 = hac_df[hai + ' W Z Score (derived)'].tolist()
    ls3 = hac_df[hai + ' W Z Score (SIS)'].tolist()
    ls4 = hac_df[hai + ' W Z Score (random)'].tolist()
    ls5 = hac_df['SSI Predicted Cases'].tolist()
    for i, val in enumerate(ls1):
        if np.isnan(val) == True: 
            #if np.isnan(ls2[i]) == False or np.isnan(ls3[i]) == False or np.isnan(ls4[i]) == False:
            #    # There are nan values for SSIs in the HACRP file that correspond to 
            #    # numeric values in the HAI file. For these facilities, the number of 
            #    # predicted cases was < 1.
            #    print(i, val)
            #    print(hai, ': ', ls2[i], '|', ls3[i], '|', ls4[i])
            #    print(ls5[i], '\n')
            ls2[i] = np.nan
            ls3[i] = np.nan
            ls4[i] = np.nan

    hac_df[hai + ' W Z Score (derived)'] = ls2
    hac_df[hai + ' W Z Score (SIS)'] = ls3
    hac_df[hai + ' W Z Score (random)'] = ls4
    
print(hac_df.shape[0], 'hospitals in hac_df')

#########################################################################################
# ##################### Get total HAC Scores #############################################
# ########################################################################################

######################## Based on derived SIR ###########################################

hac_scores = []
for hosp in hac_df['Facility ID'].tolist():
    tdf = hac_df[hac_df['Facility ID'] == hosp]
    
    w_ls = []
    sum_ls = []
    #m_ls = ['CDI W Z Score', 'CAUTI W Z Score', 'CLABSI W Z Score', 'MRSA W Z Score', 'SSI W Z Score', 'PSI-90 W Z Score']
    #m_ls = ['CDI W Z Score (derived)', 'CAUTI W Z Score (derived)', 'CLABSI W Z Score (derived)', 'MRSA W Z Score (derived)', 'SSI W Z Score', 'PSI-90 W Z Score']
    
    # Using original SSI scores because those derived from HAI files are too discrepant.
    m_ls = ['CDI W Z Score (derived)', 'CAUTI W Z Score (derived)', 
            'CLABSI W Z Score (derived)', 'MRSA W Z Score (derived)', 
            'SSI W Z Score', 'PSI-90 W Z Score']
    
    s = 0
    w = 0
    for m in m_ls:
        v = tdf[m].tolist()
        if len(list(set(v))) > 1:
            print('len(list(set(v))) > 1')
            sys.exit()
            
        v = tdf[m].iloc[0]
        
        if np.isnan(v) == False: 
            s += v
            w += 1
            
    if w > 0:
        w = 1/w
        hac_scores.append(s*w)
    else:
        w = np.nan
        hac_scores.append(np.nan)
        
hac_df['Total HAC Score (derived)'] = hac_scores
print(hac_df.shape[0], 'hospitals in hac_df')

################## Based on SIRs derived from random sampling ######################################

hac_scores = []
for hosp in hac_df['Facility ID'].tolist():
    tdf = hac_df[hac_df['Facility ID'] == hosp]
    
    w_ls = []
    sum_ls = []
    #m_ls = ['CDI W Z Score', 'CAUTI W Z Score', 'CLABSI W Z Score', 'MRSA W Z Score', 'SSI W Z Score', 'PSI-90 W Z Score']
    #m_ls = ['CDI W Z Score (derived)', 'CAUTI W Z Score (derived)', 'CLABSI W Z Score (derived)', 'MRSA W Z Score (derived)', 'SSI W Z Score', 'PSI-90 W Z Score']
    m_ls = ['CDI W Z Score (random)', 'CAUTI W Z Score (random)', 
            'CLABSI W Z Score (random)', 'MRSA W Z Score (random)', 
            'SSI W Z Score', 'PSI-90 W Z Score']
    
    s = 0
    w = 0
    for m in m_ls:
        v = tdf[m].tolist()
        if len(list(set(v))) > 1:
            print('len(list(set(v))) > 1')
            sys.exit()
            
        v = tdf[m].iloc[0]
        
        if np.isnan(v) == False: 
            s += v
            w += 1
            
    if w > 0:
        w = 1/w
        hac_scores.append(s*w)
    else:
        w = np.nan
        hac_scores.append(np.nan)
        
hac_df['Total HAC Score (random)'] = hac_scores
print(hac_df.shape[0], 'hospitals in hac_df')

########################## Based on SIS #################################################

hac_scores = []
for hosp in hac_df['Facility ID'].tolist():
    tdf = hac_df[hac_df['Facility ID'] == hosp]
    
    w_ls = []
    sum_ls = []
    m_ls = ['CDI W Z Score (SIS)', 'CAUTI W Z Score (SIS)', 
            'CLABSI W Z Score (SIS)', 'MRSA W Z Score (SIS)',
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
        
hac_df['Total HAC Score (SIS)'] = hac_scores
print(hac_df.shape[0], 'hospitals in hac_df')

#########################################################################################
# ######### Get penalty assignments based on derived HAC Scores ##########################
# ########################################################################################

######################## Based on derived SIR ###########################################

tdf = hac_df[~hac_df['Total HAC Score (derived)'].isin([np.nan, float('NaN')])]
p75 = np.percentile(tdf['Total HAC Score (derived)'], 75)

pr = []
for hosp in hac_df['Facility ID'].tolist():
    tdf = hac_df[hac_df['Facility ID'] == hosp]
    
    p = tdf['Payment Reduction'].iloc[0]
    if p != 'Yes' and p != 'No' and np.isnan(p) == True:
        pr.append(np.nan)
    
    else:
        tdf = hac_df[hac_df['Facility ID'] == hosp]
        score = tdf['Total HAC Score (derived)'].iloc[0]
        
        if np.isnan(score) == True:
            pr.append('No')
        elif score < p75:
            pr.append('No')
        elif score >= p75:
            pr.append('Yes')
        else:
            print('This score is an error:', score)
            sys.exit()

hac_df['Payment Reduction (derived)'] = pr
print(hac_df.shape[0], 'hospitals in hac_df')

######################## Based on 'randomized' SIRs ###########################################

tdf = hac_df[~hac_df['Total HAC Score (random)'].isin([np.nan, float('NaN')])]
p75 = np.percentile(tdf['Total HAC Score (random)'], 75)

pr = []
for hosp in hac_df['Facility ID'].tolist():
    tdf = hac_df[hac_df['Facility ID'] == hosp]
    
    p = tdf['Payment Reduction'].iloc[0]
    if p != 'Yes' and p != 'No' and np.isnan(p) == True:
        pr.append(np.nan)
    
    else:
        tdf = hac_df[hac_df['Facility ID'] == hosp]
        score = tdf['Total HAC Score (random)'].iloc[0]
        
        if np.isnan(score) == True:
            pr.append('No')
        elif score < p75:
            pr.append('No')
        elif score >= p75:
            pr.append('Yes')
        else:
            print('This score is an error:', score)
            sys.exit()

hac_df['Payment Reduction (random)'] = pr
print(hac_df.shape[0], 'hospitals in hac_df')

######################## Based on SIS ###########################################

tdf = hac_df[~hac_df['Total HAC Score (SIS)'].isin([np.nan, float('NaN')])]
p75 = np.percentile(tdf['Total HAC Score (SIS)'], 75)

pr = []
for hosp in hac_df['Facility ID'].tolist():
    tdf = hac_df[hac_df['Facility ID'] == hosp]
    
    p = tdf['Payment Reduction'].iloc[0]
    if p != 'Yes' and p != 'No' and np.isnan(p) == True:
        pr.append(np.nan)
    
    else:
        tdf = hac_df[hac_df['Facility ID'] == hosp]
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


hac_df['Payment Reduction (SIS)'] = pr
print(hac_df.shape[0], 'hospitals in hac_df')

#########################################################################################
# ### Check agreement of WinZ scores and total HAC scores between actual and derived #####
# ########################################################################################

print('Checking agreement of WinZ scores and total HAC scores between actual and derived:')

r = 1
threshold = 0.0075
penalty_misses = 0
hac_misses = 0
cauti_misses = 0
clabsi_misses = 0
cdiff_misses = 0
mrsa_misses = 0
ssi_misses = 0

missed_hosps = []
difs = []
hac_difs = []

hac_df = hac_df[~hac_df['Payment Reduction'].isin([np.nan, float('NaN'), 'N/A'])]

for i in range(hac_df.shape[0]):
    hosp = hac_df['Facility ID'].iloc[i]
    
    v1 = hac_df['CAUTI W Z Score'].iloc[i]
    v2 = hac_df['CAUTI W Z Score (derived)'].iloc[i]
    if np.isnan(v1) == False and np.isnan(v2) == False:
        e = err(v1, v2)
        difs.append(e)
        if e > threshold:
            print(i, 'CAUTI:', v1, v2, '\n')
            cauti_misses += 1
        
    v1 = hac_df['CLABSI W Z Score'].iloc[i]
    v2 = hac_df['CLABSI W Z Score (derived)'].iloc[i]
    if np.isnan(v1) == False and np.isnan(v2) == False:
        e = err(v1, v2)
        difs.append(e)
        if e > threshold:
            print(i, 'CLABSI:', v1, v2)
            clabsi_misses += 1
        
    v1 = hac_df['CDI W Z Score'].iloc[i]
    v2 = hac_df['CDI W Z Score (derived)'].iloc[i]
    if np.isnan(v1) == False and np.isnan(v2) == False:
        e = err(v1, v2)
        difs.append(e)
        if e > threshold:
            print(i, 'CDI:', v1, v2)
            cdiff_misses += 1
        
    v1 = hac_df['MRSA W Z Score'].iloc[i]
    v2 = hac_df['MRSA W Z Score (derived)'].iloc[i]
    if np.isnan(v1) == False and np.isnan(v2) == False:
        e = err(v1, v2)
        difs.append(e)
        if e > threshold:
            print(i, 'MRSA:', v1, v2)
            mrsa_misses += 1
    
    #v1 = hac_df['SSI W Z Score'].iloc[i]
    #v2 = hac_df['SSI W Z Score (derived)'].iloc[i]
    #if np.isnan(v1) == False and np.isnan(v2) == False:
    #    e = err(v1, v2)
    #    if e > threshold:
    #        #print(i, 'SSI:', v1, v2)
    #        ssi_misses += 1
            
    v1 = hac_df['Total HAC Score'].iloc[i]
    v2 = hac_df['Total HAC Score (derived)'].iloc[i]
    if np.isnan(v1) == False and np.isnan(v2) == False:
        e = err(v1, v2)
        if e > threshold:
            print('For CMS #', hosp, ':  Total HAC Score = ', v1, '| Total HAC Score (derived) = ', v2)
            hac_misses += 1
            missed_hosps.append(hosp)
        else:
            hac_difs.append(e)
            
    v1 = hac_df['Payment Reduction'].iloc[i]
    v2 = hac_df['Payment Reduction (derived)'].iloc[i]
    if v1 in ['Yes', 'No'] and v2 in ['Yes', 'No'] and v1 != v2:
        penalty_misses += 1

print(hac_df.shape[0], 'hospitals in final dataframe')   
print('CAUTI misses:', cauti_misses)
print('CLABSI misses:', clabsi_misses)
print('CDIFF misses:', cdiff_misses)
print('MRSA misses:', mrsa_misses)
#print('SSI misses:', ssi_misses)
print('HAC misses (due to discrepancies between HAI and HACRP files):', hac_misses)
print('Penalty misses:', penalty_misses, '\n')

print('Maximum difference between WinZ scores:', max(difs))
print('Average difference between WinZ scores:', np.mean(difs), np.std(difs), '\n')

print('Maximum difference between HAC scores:', max(hac_difs))
print('Average difference between HAC scores:', np.mean(hac_difs), np.std(hac_difs), '\n')

print("CMS numbers for hospitals with scores in the HACRP file but no data in the HAI file:", missed_hosps)

tdf = hac_df[hac_df['Total HAC Score (SIS)'] == hac_df['Total HAC Score (derived)']]
print(tdf.shape[0], ' hospitals where derived total HAC score (based on SIR) equals the total HAC score based on SIS')
tdf = hac_df[hac_df['Total HAC Score (SIS)'].isin([np.nan, float('NaN')])]
print(tdf.shape[0], ' hospitals lacking a total HAC score based on SIS')

tdf = hac_df[hac_df['Facility ID'].isin(missed_hosps)]
ls = ['PSI-90 W Z Score', 'CAUTI W Z Score', 'CDI W Z Score', 'CLABSI W Z Score', 
      'MRSA W Z Score', 'SSI W Z Score', 'Total HAC Score', 'Payment Reduction', 
      'CAUTI W Z Score (derived)', 'CDI W Z Score (derived)', 
      'CLABSI W Z Score (derived)', 'MRSA W Z Score (derived)', 
      'SSI W Z Score (derived)', 'Total HAC Score (derived)',
      ]
tdf = tdf.filter(items=ls, axis=1)

print('hac_df.shape:', hac_df.shape)
hac_df.to_pickle(mydir + 'data/yearly_compiled/HACRP-File-' + hac_mo + '-' + hac_yr + '_HAI-File-' + mo + '-' + yr + '.pkl')
