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
    'MRSA Bacteremia',
    'MRSA Bacteremia: Lower Confidence Limit',
    'MRSA Bacteremia: Observed Cases',
    'MRSA Bacteremia: Patient Days',
    'MRSA Bacteremia: Predicted Cases',
    'MRSA Bacteremia: Upper Confidence Limit',
    'MRSA Lower Confidence Limit',
    'MRSA Observed Cases',
    'MRSA Patient Days',
    'MRSA Predicted Cases',
    'MRSA Upper Confidence Limit',
    'Methicillin-resistant Staphylococcus Aureus (MRSA) Blood Laboratory-identified Events (Bloodstream infections)',
    'Methicillin-resistant Staphylococcus Aureus (MRSA) blood infections',
    ])]


d = {
     'MRSA Bacteremia': 'MRSA',
     'MRSA Bacteremia: Lower Confidence Limit': 'MRSA lower CL',
     'MRSA Bacteremia: Observed Cases': 'MRSA Observed Cases', 
     'MRSA Bacteremia: Patient Days': 'MRSA patient days',
     'MRSA Bacteremia: Predicted Cases': 'MRSA Predicted Cases', 
     'MRSA Bacteremia: Upper Confidence Limit': 'MRSA upper CL',
     'MRSA Lower Confidence Limit': 'MRSA lower CL',
     'MRSA Observed Cases': 'MRSA Observed Cases', 
     'MRSA Patient Days': 'MRSA patient days',
     'MRSA Predicted Cases': 'MRSA Predicted Cases', 
     'MRSA Upper Confidence Limit': 'MRSA upper CL',
     'Methicillin-resistant Staphylococcus Aureus (MRSA) Blood Laboratory-identified Events (Bloodstream infections)': 'MRSA',
     'Methicillin-resistant Staphylococcus Aureus (MRSA) blood infections': 'MRSA',
     
     }


measures = ['MRSA',
            'MRSA Observed Cases', 
            'MRSA patient days',
            'MRSA Predicted Cases', 
            'MRSA upper CL',
            'MRSA lower CL']


####################### Replace Names ##############################################################
print('Replace names:')
main_df['Measure Name'].replace(to_replace=d, inplace=True)
print('measure names:', list(set(main_df['Measure Name'].tolist())))
print('measures IDs:', list(set(main_df['Measure ID'].tolist())))
print('\nmain_df.shape:', main_df.shape)
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


df.to_pickle(mydir + "data/preprocessed_HAI_data/MRSA_Data.pkl")