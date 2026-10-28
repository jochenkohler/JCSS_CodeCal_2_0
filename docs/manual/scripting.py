# %% [markdown]
# # Scripting with `codecal`
#
# Everything the interface does is a few function calls. This page is the
# tour of the package: build a model, evaluate a design situation, run the
# sweep, calibrate, verify, plot, and feed a modified model back in. The
# {doc}`../reference/api` lists every function.

# %% tags=["hide-input"]
import os
import sys

import numpy as np
import pandas as pd

try:
    import codecal as cc
except ImportError:
    sys.path.insert(0, os.path.abspath('../..'))
    import codecal as cc
from codecal import plots

pd.set_option('display.width', 140)

# %% [markdown]
# ## The model
#
# `ahg_spec()` is the AHG model as a plain dictionary; `build_model`
# turns it into a `CalibrationModel` (simulating the characteristic
# loads on the way). `ahg_model()` does both. One million draws are
# enough for a script; the tests use ten million.

# %%
spec = cc.ahg_spec(n_mcs=1_000_000)
model = cc.build_model(spec)
print(model.summary()[:600], '...')

# %% [markdown]
# The model is a list of failure modes and a list of variable loads; each
# carries its random variables. A `RandomVariable` knows its
# distribution, its characteristic value and how to map a standard-normal
# `u` to `x`.

# %%
steel = model.failure_modes[0]
print(steel.name, '—', steel.R.name, type(steel.R.dist).__name__, 'CoV', round(steel.R.cov, 3), 'x_k', steel.R.char)
wind = model.loads[0]
print(wind.name, '— Q at the analysis period: mean', round(wind.Q.mean, 3), 'CoV', wind.Q.cov, 'q_k', round(wind.Q.char, 3))
print('x(u) for u = -2, 0, 2:', np.round(steel.R.dist.x_of_u(np.array([-2.0, 0.0, 2.0])), 3))

# %% [markdown]
# ## One design situation
#
# A `DesignSituation` is a failure mode, a load and the proportions
# $(a_G, a_Q)$. `design_parameter` applies the design equation,
# `limit_state` builds $g$, and `reliability` does both and runs FORM.

# %%
pf = cc.PartialFactors(gamma_G=1.35, gamma_GP=1.35, gamma_Q={'wind': 1.5, 'snow': 1.5, 'imposed': 1.5}, xi=0.85)
sit = cc.DesignSituation(steel, wind, a_G=0.8, a_Q=0.5, weight=1.0, model=model)
for eq in ('8.12', '8.13', '8.14'):
    beta, res, z = sit.reliability(pf, equation=eq)
    print(f'eq. {eq}: z = {z:.4f}, β = {beta:.3f}, converged in {res.iterations} iterations')
print('sensitivities α²:', res.sensitivities(sit.variable_names))

# %% [markdown]
# ## The sweep
#
# `evaluate` runs FORM (or SORM) for every design situation and returns
# one row per situation. `weighted_mean_beta` and `beta_table` summarise
# it; the plot functions draw it.

# %%
df = cc.evaluate(model, pf, equation='8.12')
print(len(df), 'situations; weighted-mean β =', round(cc.weighted_mean_beta(df), 3),
      '; objective M =', round(cc.objective(df, 3.8), 4))
df.head()

# %%
cc.beta_table(df).round(2)

# %%
plots.plot_beta_vs_aQ(df, 3.8)

# %% [markdown]
# ## Calibration
#
# `calibrate` minimises $M(\gamma)$ over the named parameters. Names:
# `gamma_G`, `gamma_GP`, `gamma_Q` (common), `gamma_Q:<load>` (per load),
# `xi`. Setting `gamma_G` without `gamma_GP` sets both permanent-load
# factors (the convention of the study).

# %%
res = cc.calibrate(model, free=('gamma_G', 'gamma_Q'), start=pf, equation='8.12', beta_target=3.8, verbose=False)
print(res)

# %%
plots.plot_beta_distribution(res.sweep_after, 'mode', 3.8, ref_df=res.sweep_before, labels=('calibrated', 'start'))

# %% [markdown]
# `CalibrationResult` carries the factors, the sweeps before and after,
# the optimiser history and the scipy result:

# %%
print(res.factors)
print('evaluations:', len(res.history), '| SLSQP:', res.optimizer.message)
print('γ_Q,wind of the result:', res.factors.get('gamma_Q:wind'))

# %% [markdown]
# ## Verification of one situation
#
# `form`, `sorm`, `importance_sampling` and `monte_carlo` work on any
# `LimitState` — the ones of the model, or your own.

# %%
z = sit.design_parameter(res.factors, '8.12')
ls = sit.limit_state(z)
r = cc.form(ls)
b_sorm, kappa = cc.sorm(ls, r)
isr = cc.importance_sampling(ls, r, n=50_000, seed=1)
print(f'FORM {r.beta:.4f}   SORM {b_sorm:.4f}   IS {isr.beta:.4f} ± {isr.se_beta:.4f}')

# %% [markdown]
# A limit state of your own: two variables, $g = R - S$.

# %%
R = cc.rv('R', 'lognormal', mean=2.5, cov=0.15, fractile=0.05)
S = cc.rv('S', 'gumbel', mean=1.0, cov=0.30, fractile=0.98)
ls2 = cc.LimitState(lambda x: x[0] - x[1], [R, S], name='R − S')
r2 = cc.form(ls2)
print(f'β = {r2.beta:.3f}, α = {np.round(r2.alpha, 3)}, design point x* = {np.round(r2.x_star, 3)}')
plots.plot_limit_state_2d(ls2, r2, title='R − S in u-space')

# %% [markdown]
# ## A modified model
#
# Edit the dictionary, validate, build, and compare — the same thing the
# lab mode does. Here: snow with a more uncertain model (CoV 0.26 → 0.35).

# %%
spec2 = cc.ahg_spec(n_mcs=1_000_000)
spec2['loads'][1]['theta_Q']['cov'] = 0.35
print('problems:', cc.validate_spec(spec2))
print('changes:', cc.diff_specs(spec, spec2))
model2 = cc.build_model(spec2)
df2 = cc.evaluate(model2, pf, equation='8.12')
print('weighted-mean β:', round(cc.weighted_mean_beta(df), 3), '→', round(cc.weighted_mean_beta(df2), 3))
cc.beta_table(df2).round(2)['snow'].to_frame('snow, CoV(Θ_Q) = 0.35').join(cc.beta_table(df).round(2)['snow'].to_frame('snow, CoV(Θ_Q) = 0.26'))

# %% [markdown]
# Save it for the lab mode, or for the command line
# (`python run_calibration.py --model snow_035.json`):

# %%
with open('snow_035.json', 'w', encoding='utf-8') as f:
    f.write(cc.to_json(spec2))
print(open('snow_035.json').read()[:300], '...')
os.remove('snow_035.json')
