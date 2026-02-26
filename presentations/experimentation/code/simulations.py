"""
Experiment Design Examples
Implementations of code snippets from experimentation-intro presentation.
"""

# %% setup
import numpy as np
import pandas as pd
from experiment_utils import ExperimentAnalyzer, PowerSim, balanced_random_assignment
from statsmodels.formula.api import ols

np.random.seed(42)

# %% find sample size for 3 variants, 2% absolute effect, 10% baseline, 80% power
p = PowerSim(metric='proportion',
            relative_effect=False,
            variants=2, 
            nsim=1000,
            alpha=0.05,
            alternative='two-tailed',
            comparisons=[(1, 0), (2, 0), (2, 1)],
            correction='holm',
)

result = p.find_sample_size(
    target_power=0.80,
    baseline=0.10,
    effect=[0.03, 0.05],
    # compliance=0.80,
    optimize_allocation=True,
)

# %% multiple simulation of power 
p = PowerSim(metric='proportion', 
            relative_effect=False, 
            variants=1,
            nsim=1000)

rr = p.grid_sim_power(baseline_rates=0.25,
                effects=[0.005, 0.01, 0.02, 0.03, 0.04],
                sample_sizes=[1000, 2000, 3000, 5000, 8000, 10000],
                hue='effect',
                threads=16,
                plot=True)

# %% simulate data to do sampling using balanced random assignment
n = 3000
age = np.random.normal(40, 10, n)
previous_purchases = np.random.poisson(3, n)
days_since_signup = np.random.exponential(365, n)

age_std = (age - age.mean()) / age.std()
purchases_std = (previous_purchases - previous_purchases.mean()) / previous_purchases.std()
days_std = (days_since_signup - days_since_signup.mean()) / days_since_signup.std()

logit_baseline = -2 + 2.5 * age_std + 3.0 * purchases_std - 2.0 * days_std
baseline_prob = 1 / (1 + np.exp(-logit_baseline))

df = pd.DataFrame({
    'age': age,
    'previous_purchases': previous_purchases,
    'days_since_signup': days_since_signup
})

# %% assign treatment using balanced random assignment
treatment_unblock = balanced_random_assignment(
    df,
    variants=['treatment', 'control'],
    allocation_ratio=1/2,
    balance_covariates=['age', 'previous_purchases', 'days_since_signup'],
    seed=4321
)

# %%
treatment_block = balanced_random_assignment(
    df,
    variants=['treatment', 'control'],
    allocation_ratio=1/2,
    stratification_covariates=['age', 'previous_purchases', 'days_since_signup'],
    seed=4321
)


# %% generate outcomes
is_treatment = (np.array(treatment_block) == 'treatment')
logit_with_treatment = -2 + 2.5 * age_std + 3.0 * purchases_std - 2.0 * days_std + 0.60 * is_treatment
conversion_prob = 1 / (1 + np.exp(-logit_with_treatment))
conversions = np.random.binomial(1, conversion_prob)

df['treatment'] = treatment_block
df['conversion'] = conversions

# %% simple analysis
analyzer_simple = ExperimentAnalyzer(
    df,
    treatment_col='treatment',
    outcomes='conversion',
    bootstrap=True,
    exp_sample_ratio=0.50,
    outcome_models={'conversion':['ols', 'logistic']},
    # pvalue_adjustment='sidak',
)

analyzer_simple.get_effects()
print(analyzer_simple.results.round(3)[
    ['model_type', 'absolute_effect', 'relative_effect', 'srm_pvalue']])

# %% analyze without covariate adjustment
analyzer_simple = ExperimentAnalyzer(
    df,
    treatment_col='treatment',
    outcomes='conversion',

)
analyzer_simple.get_effects()
print(analyzer_simple.results[['absolute_effect', 'pvalue', 'standard_error']])

# %% analyze with covariate adjustment
analyzer_adjusted = ExperimentAnalyzer(
    df,
    treatment_col='treatment',
    outcomes='conversion',
    regression_covariates=['age', 'previous_purchases', 'days_since_signup']
)

analyzer_adjusted.get_effects()
print(analyzer_adjusted.results[['absolute_effect', 'pvalue', 'standard_error']])
final_effect = analyzer_adjusted.results.loc[0, 'absolute_effect']

# %%
analyzer_cuped = ExperimentAnalyzer(
    df,
    treatment_col='treatment',
    outcomes='conversion',
    interaction_covariates=['age', 'previous_purchases', 'days_since_signup']
)

analyzer_cuped.get_effects()
print(analyzer_cuped.results[['absolute_effect', 'pvalue', 'standard_error']])

# %% winner curse example
np.random.seed(66)
max_iterations = 500
iteration = 0
while iteration < max_iterations:
    sample = df.sample(1000, replace=True)
    analyzer_retro = ExperimentAnalyzer(
        sample,
        treatment_col='treatment',
        outcomes='conversion'
    )
    analyzer_retro.get_effects()
    pvalue = analyzer_retro.results.loc[0, 'pvalue']
    if pvalue < 0.02:
        print(f"Significant result found at iteration {iteration} with p-value: {pvalue:.4f}")
        break
    iteration += 1

# %%
print(analyzer_retro.results[['absolute_effect', 'pvalue', 'standard_error']])
cols = ['power', 'type_s_error', 'type_m_error', 'relative_bias', 'trimmed_abs_effect']
print(f'True effect: {final_effect:.4f}')
print(analyzer_retro.calculate_retrodesign(true_effect=final_effect)[cols])

# %% checking power estimation consistency
p = PowerSim(metric='proportion',
            relative_effect=False,
            variants=1, 
            nsim=1000,
            alpha=0.05,
            alternative='two-tailed',
)
p.get_power(baseline=0.3, effect=final_effect, sample_size=500)

# %% compliance simulation
np.random.seed(42)
n = 5000

engagement = np.random.normal(0, 1, n)
account_age_months = np.random.exponential(12, n)
monthly_usage = np.random.poisson(10, n)

# random assignment (instrument Z)
assigned = np.random.binomial(1, 0.5, n)

# compliance: treatment group only, more engaged users more likely to attend
# calibrated for ~60% compliance in treatment group
compliance_prob = np.where(
    assigned == 1,
    1 / (1 + np.exp(-0.4 - 0.8 * engagement)),
    0  # control group never attends (one-sided non-compliance)
)
attended = np.random.binomial(1, compliance_prob)

print(f"Compliance rate (treatment): {attended[assigned == 1].mean():.2%}")

# %%
# outcome: bookings (revenue)
# true LATE of attending the CS meeting = 5 units
true_effect = 5
bookings = (
    50
    + 3 * engagement
    + 0.2 * account_age_months
    + 0.5 * monthly_usage
    + true_effect * attended
    + np.random.normal(0, 5, n)
)

cdf = pd.DataFrame({
    'assigned': assigned,
    'attended': attended,
    'engagement': engagement,
    'account_age_months': account_age_months,
    'monthly_usage': monthly_usage,
    'bookings': bookings,
})

# %% 
print(cdf.round(2).head())

# %% 1. Naive: compare attenders vs non-attenders (biased!)
naive = ExperimentAnalyzer(
    cdf,
    treatment_col='attended',
    outcomes='bookings',
)
naive.get_effects()
print(naive.results[['absolute_effect', 'abs_effect_lower', 'abs_effect_upper', 'pvalue']])

# %% 2. ITT: compare assigned groups (underestimates true effect)
itt = ExperimentAnalyzer(
    cdf,
    treatment_col='assigned',
    outcomes='bookings',
)
itt.get_effects()
print(itt.results[['absolute_effect', 'abs_effect_lower', 'abs_effect_upper', 'pvalue']])

# %% 3. IPW: adjust for compliance selection using covariates
ipw = ExperimentAnalyzer(
    cdf,
    treatment_col='attended',
    outcomes='bookings',
    balance_covariates=['engagement', 'account_age_months', 'monthly_usage'],
    adjustment='balance',
    balance_method='ps-logistic',
    estimand='ATT',
    overlap_plot=True
)
ipw.get_effects()
print(ipw.results[['absolute_effect', 'abs_effect_lower', 'abs_effect_upper', 'pvalue']])

# %% regression adjustment
reg_adj = ExperimentAnalyzer(
    cdf,
    treatment_col='attended',
    outcomes='bookings',
    regression_covariates=['engagement', 'account_age_months', 'monthly_usage']
)
reg_adj.get_effects()
print(reg_adj.results[['absolute_effect', 'abs_effect_lower', 'abs_effect_upper', 'pvalue']])

# %% 4. IV: Wald estimator (assignment as instrument for attendance)
# IV_wald = ITT_effect / compliance_rate
iv = ExperimentAnalyzer(
    cdf,
    treatment_col='attended',
    outcomes='bookings',
    regression_covariates=['engagement', 'account_age_months', 'monthly_usage'],
    instrument_col='assigned',
    adjustment='IV'
)
iv.get_effects()

print(iv.results[['absolute_effect', 'abs_effect_lower', 'abs_effect_upper', 'pvalue']])

# %% manual 2SLS for IV estimation
first_stage = ols('attended ~ assigned', data=cdf).fit()
cdf['predicted_attendance'] = first_stage.fittedvalues
second_stage = ols('bookings ~ predicted_attendance', data=cdf).fit()
second_stage.summary()

# %% meta-analysis example
# Experiments have different baselines correlated with allocation:
# high-alloc (many treated) -> low baseline; low-alloc (few treated) -> high baseline.
# Naive pooling conflates baseline differences with treatment effect -> inflated estimate.
np.random.seed(42)

true_effect = 0.05

experiments = [
    {"name": "exp_1", "n": 3000, "alloc": 0.20, "baseline": 0.20},
    {"name": "exp_2", "n": 2000, "alloc": 0.30, "baseline": 0.25},
    {"name": "exp_3", "n": 1500, "alloc": 0.50, "baseline": 0.30},
    {"name": "exp_4", "n": 1000, "alloc": 0.70, "baseline": 0.40},
    {"name": "exp_5", "n": 800,  "alloc": 0.80, "baseline": 0.50},
]

dfs = []
for exp in experiments:
    n = exp["n"]
    alloc = exp["alloc"]
    n_treat = int(n * alloc)
    n_control = n - n_treat
    age = np.random.normal(40, 10, n)
    treatment = np.array(["treatment"] * n_treat + ["control"] * n_control)
    np.random.shuffle(treatment)
    is_treatment = (treatment == "treatment").astype(int)
    conversion = np.random.binomial(1, exp["baseline"] + true_effect * is_treatment)
    dfs.append(pd.DataFrame({
        "experiment": exp["name"],
        "treatment": treatment,
        "conversion": conversion,
        "age": age,
    }))

meta_df = pd.concat(dfs, ignore_index=True)

# %% naive pooled analysis: ignores experiment structure -> biased
naive = ExperimentAnalyzer(
    data=meta_df,
    treatment_col="treatment",
    outcomes=["conversion"],
)
naive.get_effects()
print(naive.results[["outcome", "absolute_effect", "standard_error", "pvalue"]].round(4))

# %% correct: fixed-effects meta-analysis
analyzer = ExperimentAnalyzer(
    data=meta_df,
    treatment_col="treatment",
    outcomes=["conversion"],
    experiment_identifier="experiment",
)
analyzer.get_effects()
print(analyzer.results[["experiment", "outcome", "absolute_effect", "standard_error", "pvalue"]].round(4))

pooled = analyzer.combine_effects(grouping_cols=["outcome"])
print(pooled[["outcome", "experiments", "absolute_effect", "standard_error", "pvalue"]])
print(f"\nTrue effect: {true_effect}")

# %%
