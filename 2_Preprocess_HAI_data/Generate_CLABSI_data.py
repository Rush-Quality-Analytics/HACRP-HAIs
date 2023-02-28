import pandas as pd
import numpy as np
import sys
import warnings

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_rows', 10)
warnings.filterwarnings('ignore')

mydir = '/Users/kenlocey/GitHub/HACRP-HAIs/'

main_df = pd.read_pickle(mydir + "data/CareCompare_data/CombinedFiles_HAI/Facility.pkl")
main_df.drop(['Address', 'City', 'County Name', 'Phone Number', 'ZIP Code'], axis = 1, inplace=True)
main_df['file_date'] = main_df['file_month'] + '_' + main_df['file_year']
main_df['Facility and File Date'] = main_df['Facility ID'] + '-' + main_df['file_date']

main_df = main_df[main_df['Measure Name'].isin([
    
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Lower Confidence Limit',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Upper Confidence Limit',
    'Central Line Associated Bloodstream Infection: Number of Device Days',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Predicted Cases',
    'Central Line Associated Bloodstream Infection (ICU + select Wards): Observed Cases',
    'Central Line Associated Bloodstream Infection (ICU + select Wards)',
    
    'CLABSI: Lower Confidence Limit',
    'CLABSI: Upper Confidence Limit',
    'CLABSI: Number of Device Days',
    'CLABSI: Predicted Cases',
    'CLABSI: Observed Cases',
    'Central line-associated bloodstream infections (CLABSI) in ICUs and select wards',

    'CLABSI Lower Confidence Limit',
    'CLABSI Upper Confidence Limit',
    'CLABSI Central Line Days',
    'CLABSI Predicted Cases',
    'CLABSI Observed Cases',
    'Central line-associated blood stream infections (CLABSI) in ICUs only',
    'Central line-associated bloodstream infections (CLABSI) in ICUs only',
    
    'CLABSI: Number of Procedures',
    'CLABSI Compared to National',
    'Central line-associated blood stream infections (CLABSI)',
    'Central-Line-Associated Blood Stream Infections (CLABSI)',
    
    ])]


d = {
     'Central Line Associated Bloodstream Infection (ICU + select Wards): Lower Confidence Limit': 'CLABSI Lower Confidence Limit (ICUs + select wards)',
     'Central Line Associated Bloodstream Infection (ICU + select Wards): Upper Confidence Limit': 'CLABSI Upper Confidence Limit (ICUs + select wards)',
     'Central Line Associated Bloodstream Infection: Number of Device Days': 'CLABSI Device Days (ICUs + select wards)',
     'Central Line Associated Bloodstream Infection (ICU + select Wards): Predicted Cases': 'CLABSI Predicted Cases (ICUs + select wards)',
     'Central Line Associated Bloodstream Infection (ICU + select Wards): Observed Cases': 'CLABSI Observed Cases (ICUs + select wards)',
     'Central Line Associated Bloodstream Infection (ICU + select Wards)': 'CLABSI (ICUs + select wards)',
     
     'CLABSI: Lower Confidence Limit': 'CLABSI Lower Confidence Limit (ICUs + select wards)',
     'CLABSI: Upper Confidence Limit': 'CLABSI Upper Confidence Limit (ICUs + select wards)',
     'CLABSI: Number of Device Days': 'CLABSI Device Days (ICUs + select wards)',
     'CLABSI: Predicted Cases': 'CLABSI Predicted Cases (ICUs + select wards)',
     'CLABSI: Observed Cases': 'CLABSI Observed Cases (ICUs + select wards)',
     'Central line-associated bloodstream infections (CLABSI) in ICUs and select wards': 'CLABSI (ICUs + select wards)',

     'CLABSI Lower Confidence Limit': 'CLABSI Lower Confidence Limit (ICUs only)',
     'CLABSI Upper Confidence Limit': 'CLABSI Upper Confidence Limit (ICUs only)',
     'CLABSI Central Line Days': 'CLABSI Device Days (ICUs only)',
     'CLABSI Predicted Cases': 'CLABSI Predicted Cases (ICUs only)',
     'CLABSI Observed Cases': 'CLABSI Observed Cases (ICUs only)',
     'Central line-associated blood stream infections (CLABSI) in ICUs only': 'CLABSI (ICUs only)',
     'Central line-associated bloodstream infections (CLABSI) in ICUs only': 'CLABSI (ICUs only)',
     
     'Central line-associated blood stream infections (CLABSI)': 'CLABSI',
     'Central-Line-Associated Blood Stream Infections (CLABSI)': 'CLABSI',
     
     }

measures = [
    
     'CLABSI Lower Confidence Limit (ICUs + select wards)',
     'CLABSI Upper Confidence Limit (ICUs + select wards)',
     'CLABSI Device Days (ICUs + select wards)',
     'CLABSI Predicted Cases (ICUs + select wards)',
     'CLABSI Observed Cases (ICUs + select wards)',
     'CLABSI (ICUs + select wards)',
     
     'CLABSI Lower Confidence Limit (ICUs only)',
     'CLABSI Upper Confidence Limit (ICUs only)',
     'CLABSI Device Days (ICUs only)',
     'CLABSI Predicted Cases (ICUs only)',
     'CLABSI Observed Cases (ICUs only)',
     'CLABSI (ICUs only)',
     
     'CLABSI']


####################### Replace Names ##############################################################
print('Replace names:')
main_df['Measure Name'].replace(to_replace=d, inplace=True)
print('measure names:', list(set(main_df['Measure Name'].tolist())))
print('measures IDs:', list(set(main_df['Measure ID'].tolist())))
print('\nmain_df.shape:', main_df.shape)
IDs = list(set(main_df['Facility and File Date'].tolist()))
print(len(IDs))
print(len(list(set(main_df['Facility ID'].tolist()))))


IDs = list(set(main_df['Facility and File Date'].tolist()))
print(len(IDs))
print(len(list(set(main_df['Facility ID'].tolist()))))


measure_lists = [ [] for _ in range(len(measures)) ]
df = pd.DataFrame(columns=['Facility and File Date'])
months = []
years = []
IDs2 = []
IDs3 = []
start_dates = []
end_dates = []

for i, ID in enumerate(IDs):
    print(len(IDs) - i)
    
    tdf = main_df[main_df['Facility and File Date'] == ID]
    
    startdates = list(set(tdf['Start Date'].tolist()))
    for sd in startdates:
        tdf2 = tdf[tdf['Start Date'] == sd]
        
        months.append(tdf2['file_month'].iloc[0])
        years.append(tdf2['file_year'].iloc[0])
        start_dates.append(tdf2['Start Date'].iloc[0])
        end_dates.append(tdf2['End Date'].iloc[0])
        
        IDs2.append(ID)
        IDs3.append(tdf2['Facility ID'].iloc[0])
        
        for j, measure in enumerate(measures):
            tdf3 = 0
            try:
                tdf3 = tdf2[tdf2['Measure Name'] == measure]
                val = tdf3['Score'].iloc[0]
            except:
                val = np.nan
                
            measure_lists[j].append(val)
        
df['Facility and File Date'] = IDs2
df['Facility ID'] = IDs3
df['Start Date'] = start_dates
df['End Date'] = end_dates

for i, measure in enumerate(measures):
    df[measure] = measure_lists[i]
    
    
df['file_month'] = months
df['file_year'] = years
df.drop(labels=['Facility and File Date'], axis=1, inplace=True)

df.to_pickle(mydir + "data/preprocessed_HAI_data/CLABSI_Data.pkl")
