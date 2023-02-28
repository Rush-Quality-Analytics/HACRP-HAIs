import HAI_optimize

obs_days = 'CLABSI Device Days'
pred_cases = 'CLABSI Predicted Cases'
obs_cases = 'CLABSI Observed Cases'
hai = 'CLABSI'
z_ran = [1, 10000]
pi_ran = [0.0001, 0.01]

HAI_optimize.optimize(obs_days, pred_cases, obs_cases, hai, z_ran, pi_ran)
