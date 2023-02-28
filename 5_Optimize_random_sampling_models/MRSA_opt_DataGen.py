import HAI_optimize

obs_days = 'MRSA patient days'
pred_cases = 'MRSA Predicted Cases'
obs_cases = 'MRSA Observed Cases'
hai = 'MRSA'
z_ran = [10000, 200000]
pi_ran = [0.00001, 0.001]

HAI_optimize.optimize(obs_days, pred_cases, obs_cases, hai, z_ran, pi_ran)
