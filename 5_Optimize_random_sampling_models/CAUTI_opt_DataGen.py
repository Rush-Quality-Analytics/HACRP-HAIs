import HAI_optimize

obs_days = 'CAUTI Urinary Catheter Days'
pred_cases = 'CAUTI Predicted Cases'
obs_cases = 'CAUTI Observed Cases'
hai = 'CAUTI'
z_ran = [1, 20000]
pi_ran = [0.0001, 0.01]

HAI_optimize.optimize(obs_days, pred_cases, obs_cases, hai, z_ran, pi_ran)
