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
#print(main_df.head())
#sys.exit()

#main_df = main_df[(main_df['file_year'] == '2021') & (main_df['file_month'] == '04')]
main_df['file_date'] = main_df['file_month'] + '_' + main_df['file_year']
main_df['Facility and File Date'] = main_df['Facility ID'] + '-' + main_df['file_date']

#measures = sorted(list(set(main_df['Measure Name'].tolist())))
#print(measures)

SSI_measures = ['SSI - Colon Surgery', 
                'SSI - Colon Surgery: Lower Confidence Limit', 
                'SSI - Colon Surgery: Number of Procedures', 
                'SSI - Colon Surgery: Observed Cases', 
                'SSI - Colon Surgery: Predicted Cases', 
                'SSI - Colon Surgery: Upper Confidence Limit', 
                'SSI: Colon Lower Confidence Limit', 
                'SSI: Colon Observed Cases', 
                'SSI: Colon Predicted Cases', 
                'SSI: Colon Upper Confidence Limit', 
                'SSI: Colon, Number of Procedures', 
                'Surgical Site Infection from colon surgery (SSI: Colon)', 
                'Surgical site infections (SSI) from colon surgery',
                
                'SSI - Abdominal Hysterectomy', 
                'SSI - Abdominal Hysterectomy: Lower Confidence Limit', 
                'SSI - Abdominal Hysterectomy: Number of Procedures', 
                'SSI - Abdominal Hysterectomy: Observed Cases', 
                'SSI - Abdominal Hysterectomy: Predicted Cases', 
                'SSI - Abdominal Hysterectomy: Upper Confidence Limit', 
                'SSI: Abdominal Lower Confidence Limit', 
                'SSI: Abdominal Observed Cases', 
                'SSI: Abdominal Predicted Cases', 
                'SSI: Abdominal Upper Confidence Limit', 
                'SSI: Abdominal, Number of Procedures', 
                'Surgical Site Infection from abdominal hysterectomy (SSI: Hysterectomy)', 
                'Surgical site infections (SSI) from abdominal hysterectomy', 
                ]

main_df = main_df[main_df['Measure Name'].isin(SSI_measures)]

d = {'Surgical Site Infection from colon surgery (SSI: Colon)': 'SSI-CP',
     'SSI - Colon Surgery': 'SSI-CP',
     'Surgical site infections (SSI) from colon surgery':  'SSI-CP',
     'SSI - Colon Surgery: Number of Procedures': 'SSI: Colon, Number of Procedures', 
     'SSI - Colon Surgery: Observed Cases': 'SSI: Colon Observed Cases', 
     'SSI - Colon Surgery: Predicted Cases': 'SSI: Colon Predicted Cases', 
     
     'Surgical Site Infection from abdominal hysterectomy (SSI: Hysterectomy)': 'SSI-AH',
     'SSI - Abdominal Hysterectomy': 'SSI-AH',
     'Surgical site infections (SSI) from abdominal hysterectomy': 'SSI-AH',
     'SSI - Abdominal Hysterectomy: Number of Procedures': 'SSI: Abdominal, Number of Procedures',
     'SSI - Abdominal Hysterectomy: Observed Cases': 'SSI: Abdominal Observed Cases', 
     'SSI - Abdominal Hysterectomy: Predicted Cases': 'SSI: Abdominal Predicted Cases', 
     }

measures = ['SSI: Colon, Number of Procedures', 
            'SSI-CP', 
            'SSI: Colon Observed Cases', 
            'SSI: Colon Predicted Cases',
            
            'SSI: Abdominal, Number of Procedures',
            'SSI-AH', 
            'SSI: Abdominal Observed Cases', 
            'SSI: Abdominal Predicted Cases'
            ]
measure_lists = [ [] for _ in range(len(measures)) ]


main_df['Measure Name'].replace(to_replace=d, inplace=True)
print(list(set(main_df['Measure Name'].tolist())))


df = pd.DataFrame(columns=['Facility and File Date'])
IDs = list(set(main_df['Facility and File Date'].tolist()))

months = []
years = []
IDs2 = []
IDs3 = []
start_dates = []
end_dates = []

for i, ID in enumerate(IDs):
    print(len(IDs) - i)
    tdf = main_df[main_df['Facility and File Date'] == ID]
    
    sd = list(set(tdf['Start Date'].tolist()))
    if len(sd) > 1:
        print(tdf.head(tdf.shape[0]))
        print(sd)
        sys.exit()
    
    months.append(tdf['file_month'].iloc[0])
    years.append(tdf['file_year'].iloc[0])
    start_dates.append(tdf['Start Date'].iloc[0])
    end_dates.append(tdf['End Date'].iloc[0])
    
    IDs2.append(ID)
    IDs3.append(tdf['Facility ID'].iloc[0])
    
    for j, measure in enumerate(measures):
        tdf2 = 0
        try:
            tdf2 = tdf[tdf['Measure Name'] == measure]
            val = tdf2['Score'].iloc[0]
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

df['SSI: Colon Observed Cases'] = pd.to_numeric(df['SSI: Colon Observed Cases'], errors='coerce')
df['SSI: Abdominal Observed Cases'] = pd.to_numeric(df['SSI: Abdominal Observed Cases'], errors='coerce')
df['SSI: Colon Predicted Cases'] = pd.to_numeric(df['SSI: Colon Predicted Cases'], errors='coerce')
df['SSI: Abdominal Predicted Cases'] = pd.to_numeric(df['SSI: Abdominal Predicted Cases'], errors='coerce')
df['SSI: Abdominal, Number of Procedures'] = pd.to_numeric(df['SSI: Abdominal, Number of Procedures'], errors='coerce')
df['SSI: Colon, Number of Procedures'] = pd.to_numeric(df['SSI: Colon, Number of Procedures'], errors='coerce')

df['SSI Observed Cases'] = df[['SSI: Colon Observed Cases', 'SSI: Abdominal Observed Cases']].sum(axis=1, min_count=1)
df['SSI Predicted Cases'] = df[['SSI: Colon Predicted Cases', 'SSI: Abdominal Predicted Cases']].sum(axis=1, min_count=1)
df['SSI Number of Procedures'] = df[['SSI: Abdominal, Number of Procedures', 'SSI: Colon, Number of Procedures']].sum(axis=1, min_count=1)
df['SSI'] = df['SSI Observed Cases']/df['SSI Predicted Cases']

tdf = df.filter(items=['SSI: Colon Observed Cases', 'SSI: Colon Predicted Cases',
                       'SSI: Abdominal Observed Cases', 'SSI: Abdominal Predicted Cases',
                       'SSI Observed Cases', 'SSI Predicted Cases',
                       ], axis=1)

df.to_pickle(mydir + "1_preprocess_CareCompare_data/preprocessed_HAI_data/SSI_Data.pkl")
