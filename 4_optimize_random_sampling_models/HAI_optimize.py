import pandas as pd
import numpy as np
import random
import sys
import os
import scipy as sc
import warnings
from scipy.stats import sem
from scipy import stats

np.random.seed(0)
random.seed(0)

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)
warnings.filterwarnings('ignore')

mydir = '/Users/kenlocey/GitHub/HACRP-HAIs/'

def obs_pred_rsquare(obs, pred):
    obs = np.sqrt(obs)
    pred = np.sqrt(pred)
    # Determines the prop of variability in a data set accounted for by a model
    # In other words, this determines the proportion of variation explained by
    # the 1:1 line in an observed-predicted plot.
    return 1 - sum((obs - pred) ** 2) / sum((obs - np.mean(obs)) ** 2)


def optimize(volume, pred_cases, obs_cases, hai, z_ran, pi_ran):

    main_df = pd.read_pickle(mydir + "data/Compiled_HCRIS-HACRP-HAI-RAND/Compiled_HCRIS-HACRP-HAI-RAND.pkl")
    
    main_df['file date'] = main_df['file_year'] + '_' + main_df['file_month']
    fdates = sorted(list(set(main_df['file date'].tolist())))
    
    main_df = main_df[~main_df[obs_cases].isin([np.nan, 'Not Available'])]
    main_df = main_df[~main_df[pred_cases].isin([np.nan, 'Not Available'])]
    main_df = main_df[~main_df[volume].isin([np.nan, 'Not Available'])]

    main_df[obs_cases] = main_df[obs_cases].astype(float)
    main_df[pred_cases] = main_df[pred_cases].astype(float)
    main_df = main_df[main_df[pred_cases] >= 1]

    main_df[volume] = main_df[volume].astype('int64')
    main_df['O/E'] = main_df[obs_cases] / main_df[pred_cases]
    main_df['simulated O'] = [np.nan] * main_df.shape[0]
    main_df['simulated O/E'] = [np.nan] * main_df.shape[0]

    main_df['expected O'] = [np.nan] * main_df.shape[0]
    main_df['expected O/E'] = [np.nan] * main_df.shape[0]
    main_df['pi_opt'] = [np.nan] * main_df.shape[0]
    main_df['z_opt'] = [np.nan] * main_df.shape[0]


    for fdate in fdates:
        print(fdate)
        
        pi_opt = 0
        pi = 0

        z_opt = 0
        z = 0

        pi_opt_ls = []
        z_opt_ls = []
        avg_pval_ls = []
        se_pval_ls = []
        std_pval_ls = []
        avg_r2_ls = []
        ct_ls = []

        simulated_cases_opt = []
        expected_cases_opt = []

        pval_opt = 0
        std_pval_opt = 0
        se_pval_opt = 0
        r2_opt = 0
        ct = 0
        
        df = main_df[main_df['file date'] == fdate]
        print('rows:', df.shape[0])
        if df.shape[0] < 100:
            continue
        
        vol = np.array(df[volume].tolist())
        predicted_cases = np.array(df[pred_cases].tolist())
        observed_cases = np.array(df[obs_cases].tolist())

        observed_SIR = observed_cases/predicted_cases
        observed_SIR = observed_SIR.tolist()
        
        while ct < 5*10**3:
            
            ct += 1
            if ct < 2500:
                # choose pi and z based on uniform random sampling
                pi = np.random.uniform(min(pi_ran), max(pi_ran))
                z = np.random.uniform(min(z_ran), max(z_ran))

            else:
                max_avg_pval = max(avg_pval_ls)
                i = avg_pval_ls.index(max_avg_pval)
                
                pi = np.abs(np.random.normal(pi_opt_ls[i], 0.001))
                z = np.abs(np.random.normal(z_opt_ls[i], 10))
            
            pD = vol/(z + vol)
            p = pi * pD
            
            pval_ls1 = []
            r2_ls1 = []
            
            iter = 100
            for i in range(iter):
                
                simulated_cases = np.array(np.random.binomial(vol, p=p, size=len(vol)))
                r2 = obs_pred_rsquare(np.array(observed_cases), np.array(simulated_cases))
                stat, c_vals, p_val = stats.anderson_ksamp(np.array([simulated_cases, observed_cases]))
                pval_ls1.append(p_val)
                r2_ls1.append(r2)
            
            sim_pval = np.nanmean(pval_ls1)
            std_pval = np.std(pval_ls1)
            se_pval = sem(pval_ls1)
            
            expected_cases = p * vol
            exp_r2 = obs_pred_rsquare(np.array(observed_cases), np.array(expected_cases))
            stat, c_vals, exp_pval = stats.anderson_ksamp(np.array([expected_cases, observed_cases]))
            
            if ct == 1 or (sim_pval > pval_opt) or (sim_pval >= pval_opt and exp_r2 > r2_opt):
            
                pi_opt = float(pi)
                z_opt = float(z)
                pval_opt = float(sim_pval)
                std_pval_opt = float(std_pval)
                se_pval_opt = float(se_pval)
                r2_opt = float(exp_r2)
                
                vol = np.array(df[volume].tolist())
                pD = vol/(z_opt + vol)
                p = pi_opt * pD
                simulated_cases_opt = np.array(np.random.binomial(vol, p=p, size=len(vol)))
                expected_cases_opt = p * vol
                
            if ct == 1 or ct%500 == 0:
                print(ct)
                print('pi_opt:', pi_opt, '   |   z_opt:', z_opt)
                print('avg. p-val: ', np.round(pval_opt, 5), '   |   r2 (obs vs exp): ', np.round(r2_opt, 5), '\n')
                
                pi_opt_ls.append(pi_opt)
                z_opt_ls.append(z_opt)
                avg_pval_ls.append(pval_opt)
                std_pval_ls.append(std_pval_opt)
                se_pval_ls.append(se_pval_opt)
                avg_r2_ls.append(r2_opt)
                ct_ls.append(ct)
                
                df['simulated O'] = simulated_cases_opt
                df['simulated O/E'] = df['simulated O'] / df[pred_cases]
                    
                df['expected O'] = expected_cases_opt
                df['expected O/E'] = df['expected O'] / df[pred_cases]
                    
                df['pi_opt'] = [pi_opt]*len(simulated_cases_opt)
                df['z_opt'] = [z_opt]*len(simulated_cases_opt)
            
                opt_df = pd.DataFrame(columns=['iteration'])
                opt_df['iteration'] = ct_ls
                opt_df['avg_pval'] = avg_pval_ls
                opt_df['std_pval'] = std_pval_ls
                opt_df['se_pval'] = se_pval_ls
                opt_df['r2 (obs vs exp)'] = avg_r2_ls
                opt_df['pi_opt'] = pi_opt_ls
                opt_df['z_opt'] = z_opt_ls
                
                df.to_pickle(mydir + "4_optimize_random_sampling_models/optimized_by_HAI_file_date/" + hai + "/" + hai + "_Data_opt_for_SIRs_" + fdate + ".pkl")
                opt_df.to_csv(mydir + "4_optimize_random_sampling_models/optimized_by_HAI_file_date/" + hai + "/" + hai + "_opt_iterations_" + fdate + ".csv")
                
                
        print('Finished:', fdate)

