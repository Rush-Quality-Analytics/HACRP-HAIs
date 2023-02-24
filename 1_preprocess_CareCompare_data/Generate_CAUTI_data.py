import pandas as pd
import numpy as np
import sys
import warnings

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = '/Users/kenlocey/GitHub/HACRP-HAIs/'

main_df = pd.read_pickle(mydir + "data/CareCompare_data/CombinedFiles_HAI/Facility.pkl")
main_df.drop(['Address', 'City', 'County Name', 'Phone Number', 'ZIP Code'], axis = 1, inplace=True)
#print(list(main_df))
#print(main_df.head(), '\n')

#for i in main_df['Measure Name'].unique(): print(i)
#sys.exit()

main_df['file_date'] = main_df['file_month'] + '_' + main_df['file_year']
main_df['Facility and File Date'] = main_df['Facility ID'] + '-' + main_df['file_date']

main_df = main_df[main_df['Measure Name'].isin([
    
    'CAUTI Lower Confidence Limit',
    'CAUTI Upper Confidence Limit',
    'CAUTI Urinary Catheter Days',
    'CAUTI Predicted Cases',
    'CAUTI Observed Cases',
    'Catheter-Associated Urinary Tract Infections (CAUTI) in ICUs only',
    
    'CAUTI: Lower Confidence Limit',
    'CAUTI: Upper Confidence Limit',
    'CAUTI: Number of Urinary Catheter Days',
    'CAUTI: Predicted Cases',
    'CAUTI: Observed Cases',
    'Catheter-associated urinary tract infections (CAUTI) in ICUs and select wards',
    
    'Catheter Associated Urinary Tract Infections (ICU + select Wards): Lower Confidence Limit',
    'Catheter Associated Urinary Tract Infections (ICU + select Wards): Upper Confidence Limit',
    'Catheter Associated Urinary Tract Infections (ICU + select Wards): Number of Urinary Catheter Days',
    'Catheter Associated Urinary Tract Infections (ICU + select Wards): Predicted Cases',
    'Catheter Associated Urinary Tract Infections (ICU + select Wards): Observed Cases',
    'Catheter Associated Urinary Tract Infections (ICU + select Wards)',
    
    'CAUTI: Number of Procedures',
    'CAUTI Compared to National',
    'Catheter-Associated Urinary Tract Infections (CAUTI)',
    
    ])]


d = {
     'CAUTI Lower Confidence Limit': 'CAUTI Lower Confidence Limit (ICUs only)',
     'CAUTI Upper Confidence Limit': 'CAUTI Upper Confidence Limit (ICUs only)',
     'CAUTI Urinary Catheter Days': 'CAUTI Urinary Catheter Days (ICUs only)',
     'CAUTI Predicted Cases': 'CAUTI Predicted Cases (ICUs only)',
     'CAUTI Observed Cases': 'CAUTI Observed Cases (ICUs only)',
     'Catheter-Associated Urinary Tract Infections (CAUTI) in ICUs only': 'CAUTI (ICUs only)',
     
     'CAUTI: Lower Confidence Limit': 'CAUTI Lower Confidence Limit (ICUs + select wards)',
     'CAUTI: Upper Confidence Limit': 'CAUTI Upper Confidence Limit (ICUs + select wards)',
     'CAUTI: Number of Urinary Catheter Days': 'CAUTI Urinary Catheter Days (ICUs + select wards)',
     'CAUTI: Predicted Cases': 'CAUTI Predicted Cases (ICUs + select wards)',
     'CAUTI: Observed Cases': 'CAUTI Observed Cases (ICUs + select wards)',
     'Catheter-associated urinary tract infections (CAUTI) in ICUs and select wards': 'CAUTI (ICUs + select wards)',
     
     'Catheter Associated Urinary Tract Infections (ICU + select Wards): Lower Confidence Limit': 'CAUTI Lower Confidence Limit (ICUs + select wards)',
     'Catheter Associated Urinary Tract Infections (ICU + select Wards): Upper Confidence Limit': 'CAUTI Upper Confidence Limit (ICUs + select wards)',
     'Catheter Associated Urinary Tract Infections (ICU + select Wards): Number of Urinary Catheter Days': 'CAUTI Urinary Catheter Days (ICUs + select wards)',
     'Catheter Associated Urinary Tract Infections (ICU + select Wards): Predicted Cases': 'CAUTI Predicted Cases (ICUs + select wards)',
     'Catheter Associated Urinary Tract Infections (ICU + select Wards): Observed Cases': 'CAUTI Observed Cases (ICUs + select wards)',
     'Catheter Associated Urinary Tract Infections (ICU + select Wards)': 'CAUTI (ICUs + select wards)',
     
     'Catheter-Associated Urinary Tract Infections (CAUTI)': 'CAUTI',
     
}

measures = [
     'CAUTI Lower Confidence Limit (ICUs only)',
     'CAUTI Upper Confidence Limit (ICUs only)',
     'CAUTI Urinary Catheter Days (ICUs only)',
     'CAUTI Predicted Cases (ICUs only)',
     'CAUTI Observed Cases (ICUs only)',
     'CAUTI (ICUs only)',
     
     'CAUTI Lower Confidence Limit (ICUs + select wards)',
     'CAUTI Upper Confidence Limit (ICUs + select wards)',
     'CAUTI Urinary Catheter Days (ICUs + select wards)',
     'CAUTI Predicted Cases (ICUs + select wards)',
     'CAUTI Observed Cases (ICUs + select wards)',
     'CAUTI (ICU + select wards)',
     
     'CAUTI']

####################### Replace Names ##############################################################
print('Replace names:')
main_df['Measure Name'].replace(to_replace=d, inplace=True)
print('measure names:', list(set(main_df['Measure Name'].tolist())))
print('measures IDs:', list(set(main_df['Measure ID'].tolist())))
print('\nmain_df.shape:', main_df.shape)
IDs = list(set(main_df['Facility and File Date'].tolist()))
print(len(IDs))
print(len(list(set(main_df['Facility ID'].tolist()))))


'''
####################### Filter Measures ############################################################
print('Filter measure names:')
measures = ['CAUTI Number of Procedures', 'CAUTI', 'CAUTI upper CL', 
            'CAUTI lower CL', 'CAUTI Urinary Catheter Days', 
            'CAUTI Observed Cases', 'CAUTI Predicted Cases']
main_df = main_df[main_df['Measure Name'].isin(measures)]
print('\nmain_df.shape:', main_df.shape)
IDs = list(set(main_df['Facility and File Date'].tolist()))
print(len(IDs))
print(len(list(set(main_df['Facility ID'].tolist()))))
'''

'''
####################### Filter Measure IDs #########################################################
print('Filter measure IDs:')
main_df = main_df[~main_df['Measure ID'].isin(['HAI_2a_DOPC_DAYS', 'HAI_2a_CI_UPPER', 
                                               'HAI_2a_NUMERATOR', 'HAI_2a_ELIGCASES', 
                                               'HAI_2a_CI_LOWER'])]
print('\nmain_df.shape:', main_df.shape)
'''

IDs = list(set(main_df['Facility and File Date'].tolist()))
#print(len(IDs))
#print(len(list(set(main_df['Facility ID'].tolist()))))


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


df.to_pickle(mydir + "1_preprocess_CareCompare_data/preprocessed_HAI_data/CAUTI_Data.pkl")

