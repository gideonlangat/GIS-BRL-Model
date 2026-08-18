# GIS-BRL
This code implement simulation of the GIS-BRL (GIS-Enhanced Bayesian Reinforced Learning)

Implements, end to end, the model developed in Chapter 3 (as revised):
location-and-wave-indexed Bayesian hierarchical risk estimation with a
non-reversible-style MH / Gibbs sampler, feeding a location-conditioned,
five-critic Min-Max ensemble TD3 policy that learns prescriptive,
location-adaptive intervention rates across the 8 study counties.

## File:GIS-BRL.ipynb

All hyperparameters: county profiles, sampler settings,
RL settings, environment dynamics, Data simulation,
non-reversible-style MH sampler for `beta_j`,
Gibbs sampler for `sigma2_j`, exact conjugate form),
MH for `u_i` and `gamma_t`, convergence diagnostics (Gelman-Rubin, ESS,
autocorrelation, MCSE), posterior predictive checks,
and `fit_pi_it`,
DiseaseSpreadEnv: the location-conditioned host-vector interaction environment. Interaction is the case-generating event (interaction = case), modulated by intervention rates and county risk `pi_it`,
Actor + 5 critics, min-max ensemble target, TD3 training step,
Training loop, evaluation (success/infection rate), learned-rate extraction, predicted case counts, case prediction error, cross-location adaptivity variance, Runs the full pipeline and writes everything to `outputs/`

Load

pip install numpy pandas scipy matplotlib torch
python3 main.py

N_PER_COUNTY_WAVE = 1500   
N_ITER = 10000              
N_TRAIN_EPISODES = 5000     
CASE_EVAL_STEPS = 20000    

Runtime scales roughly linearly with these; expect a full-scale run to
take on the order of hours on CPU (the RL stage dominates: each
environment step trains 5 critic networks). A GPU is not required at
this network size but will help if you increase hidden layer width.

## Outputs (written to `outputs/`)

Bayesian estimation

- `trace_plots_beta.png` — trace plots for all `beta_j`, 3 chains, burn-in shaded, true value marked (as requested).
- `beta_convergence_diagnostics.csv` / `gelman_rubin.png` — Gelman-Rubin R-hat, ESS, MCSE, lag-1 autocorrelation per parameter.
- `beta_estimation_accuracy.png`, `beta_rmse.png` — posterior mean vs. true beta, per-parameter and overall RMSE.
- `posterior_predictive_check.png` — replicated vs. observed case proportion, Bayesian p-value.
- `fitted_pi_it.csv` — the fitted, location-and-wave-varying risk surface used to shape RL rewards.

Design notes / assumptions made explicit in code

- Reward cost term is quadratic 
- Environment scale is normalized
- Case Prediction Error is large
- sigma2_j sampling is exact Gibbs
- Sum-to-zero identifiability constraint, on `u_i` and `gamma_t`
- Spraying and net-use are given distinct protective effects and costs
- wealth_index and education_level are dummy-coded, not treated as continuous.
- Infection rate metric was redefined from per-episode to per-time-step.
- The policy was collapsing to literal 0%/100% intervention rates.
- Recalibrated to the actual real observed case counts, and made cases visibly decline as the policy is learned.
