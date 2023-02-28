import HAI_optimize

obs_days = 'CDI patient days'
pred_cases = 'CDI Predicted Cases'
obs_cases = 'CDI Observed Cases'
hai = 'CDI'
z_ran = [1, 40000]
pi_ran = [0.00001, 0.01]

HAI_optimize.optimize(obs_days, pred_cases, obs_cases, hai, z_ran, pi_ran)
