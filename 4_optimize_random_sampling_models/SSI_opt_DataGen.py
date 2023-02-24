import HAI_optimize

num_procedures = 'SSI Number of Procedures'
pred_cases = 'SSI Predicted Cases'
obs_cases = 'SSI Observed Cases'
hai = 'SSI'
z_ran = [1, 1000]
pi_ran = [0.001, 0.1]

HAI_optimize.optimize(num_procedures, pred_cases, obs_cases, hai, z_ran, pi_ran)
