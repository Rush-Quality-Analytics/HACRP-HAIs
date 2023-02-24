import HAI_optimize

num_procedures = 'SSI: Abdominal, Number of Procedures'
pred_cases = 'SSI: Abdominal Predicted Cases'
obs_cases = 'SSI: Abdominal Observed Cases'
hai = 'SSI-AH'
z_ran = [1, 1000]
pi_ran = [0.001, 0.1]

HAI_optimize.optimize(num_procedures, pred_cases, obs_cases, hai, z_ran, pi_ran)
