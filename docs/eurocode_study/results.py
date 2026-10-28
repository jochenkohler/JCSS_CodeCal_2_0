# %% [markdown]
# # Results of the Eurocode study — reproduced
#
# This page reruns the reliability benchmark and the calibrations of the
# AHG study with CodeCal 2.0 and puts the numbers next to those reported
# in the AHG working document {cite}`KoehlerHingoraniSorensen2021` and in
# {cite}`Kohler2023,KohlerChapter2026`. Everything below is computed when
# the notebook is executed; nothing is typed in by hand except the
# reference values.
#
# All reliability indices are FORM values for a 50-year reference period,
# as in the study, on the grid of 10 values of $a_Q$ × 3 values of $a_G$ per
# failure mode (810 design situations) that the study's code used. The
# {doc}`verification` page shows what SORM and sampling say about the FORM
# error.

# %% tags=["hide-input"]
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import codecal as cc
except ImportError:
    sys.path.insert(0, os.path.abspath('../..'))
    import codecal as cc
from codecal import plots

pd.set_option('display.width', 140)
pd.set_option('display.precision', 3)

t0 = time.time()
model = cc.ahg_model()                       # 10 M Monte-Carlo draws for the characteristic loads, seed 12345
print(model.summary())
print(f'model built in {time.time() - t0:.1f} s')

# %% [markdown]
# ## Reference values from the AHG study
#
# Weighted-average 50-year reliability indices with the present factors
# ($\gamma_G = 1.35$, $\gamma_Q = 1.5$, $\xi = 0.85$, $\psi_0$ = 0.6 / 0.5 /
# 0.7), and the calibrated factors for $\beta_t = 3.8$, as reported in
# the November 2021 working document (design equations "6.10" and
# "6.10 a/b" of EN 1990:2002, which became (8.12) and (8.13) of
# EN 1990:2023) and in the 2026 chapter (which adds (8.14)).

# %%
AHG = {
    'status quo': {'8.12': 3.74, '8.13': 3.44, '8.14': None},         # weighted averages of the Nov 2021 document (no 8.14 there)
    'status quo, chapter': {'8.12': 3.74, '8.13': 3.51, '8.14': 3.48},  # as printed in the 2026 chapter (see the note below)
    'crude 8.12': 3.65,
    'cal 8.12, common γ_Q':  {'gamma_G': 1.22, 'gamma_Q': 1.75, 'beta': 3.81},
    'cal 8.12, per load':    {'gamma_G': 1.19, 'gamma_Q:wind': 1.94, 'gamma_Q:snow': 2.35, 'gamma_Q:imposed': 1.60, 'beta': 3.81},
    'cal 8.13, common γ_Q':  {'xi': 0.89, 'gamma_G': 1.33, 'gamma_Q': 1.78, 'beta': 3.80},
    'cal 8.13, per load':    {'xi': 0.81, 'gamma_G': 1.33, 'gamma_Q:wind': 2.05, 'gamma_Q:snow': 2.52, 'gamma_Q:imposed': 1.69, 'beta': 3.86},
    'std 8.12': {'status quo': 0.46, 'calibrated per load': 0.24},
}
AHG_TABLE_812 = pd.DataFrame(                 # "Average reliability indices per material and action", eq. 6.10
    [[3.52, 3.17, 3.94], [3.84, 3.62, 4.15], [3.42, 3.15, 3.78], [3.34, 3.08, 3.71], [3.41, 3.17, 3.74],
     [4.10, 3.92, 4.37], [3.61, 3.31, 4.00], [3.55, 3.42, 3.79], [3.73, 3.59, 3.98]],
    index=[m.name for m in model.failure_modes], columns=['wind', 'snow', 'imposed'])

# %% [markdown]
# ## 1. The status quo: reliability with the present partial factors
#
# The same 810 design situations, designed with each of the three
# combination rules of EN 1990:2023 and evaluated with FORM.

# %%
pf0 = cc.START_VALUES
sweeps = {}
for eq in ('8.12', '8.13', '8.14'):
    t0 = time.time()
    sweeps[eq] = cc.evaluate(model, pf0, equation=eq)
    print(f'eq. {eq}: {len(sweeps[eq])} situations, {time.time() - t0:.1f} s')

def stats(df):
    w = df['weight']
    wm = cc.weighted_mean_beta(df)
    return {'weighted mean': wm, 'crude mean': df['beta'].mean(),
            'weighted std': float(np.sqrt((w * (df['beta'] - wm) ** 2).sum())),
            'std': df['beta'].std(), 'min': df['beta'].min(), 'max': df['beta'].max(),
            'M(γ) = Σw(β−3.8)²': cc.objective(df, 3.8)}

overview = pd.DataFrame({eq: stats(df) for eq, df in sweeps.items()}).T
overview.insert(0, 'AHG 2021, weighted mean', [AHG['status quo'][eq] for eq in overview.index])
overview.insert(1, 'chapter 2026, weighted mean', [AHG['status quo, chapter'][eq] for eq in overview.index])
overview

# %% [markdown]
# The weighted averages reproduce the 2021 document: 3.74 for (8.12) and
# 3.44 for (8.13) — then called 6.10 and 6.10 a/b. (8.13) and (8.14) land a
# quarter to a third of a β-unit lower than (8.12), because the second
# combination (ξ on the permanent load) governs most situations and gives
# a smaller design value. The scatter is as reported — a standard
# deviation of about 0.4–0.46 for (8.12), somewhat less for the
# two-combination rules.
#
# ```{note}
# The chapter manuscript prints 3.51 for (8.13) and 3.48 for (8.14). Those
# two values are reproduced here with ξ = 0.89 — the value that the 2021
# calibration of (8.13) returned — instead of the recommended ξ = 0.85
# (next cell). The table of the chapter was apparently produced with the
# calibrated ξ left in the start values; the 2021 document and the
# recommended value of EN 1990 give 3.44 and 3.38.
# ```

# %%
pf089 = cc.PartialFactors(1.35, 1.35, {'wind': 1.5, 'snow': 1.5, 'imposed': 1.5}, xi=0.89)
pd.DataFrame({eq: {'ξ = 0.85 (EN 1990)': cc.weighted_mean_beta(sweeps[eq]),
                   'ξ = 0.89': cc.weighted_mean_beta(cc.evaluate(model, pf089, equation=eq)),
                   'chapter 2026': AHG['status quo, chapter'][eq]} for eq in ('8.13', '8.14')}).T

# %%
fig = plots.plot_beta_distribution(sweeps['8.12'], 'mode', 3.8, title='Status quo, eq. 8.12 — β per failure mode')
fig

# %%
from codecal.plots import Figure
fig = Figure(figsize=(10, 4.2))
axes = fig.subplots(1, 3, sharey=True)
for ax, (eq, df) in zip(axes, sweeps.items()):
    for load, grp in df.groupby('load'):
        g = grp.groupby('a_Q')['beta']
        ax.plot(g.mean().index, g.mean().values, '-', label=load)
        ax.fill_between(g.min().index, g.min().values, g.max().values, alpha=0.15)
    ax.axhline(3.8, color=plots.TARGET, ls='--', lw=1)
    ax.set_title(f'eq. {eq}: weighted mean {cc.weighted_mean_beta(df):.2f}', fontsize=10)
    ax.set_xlabel('a_Q (share of the variable load)')
    ax.grid(alpha=0.3)
axes[0].set_ylabel('β (50 a)')
axes[0].legend(fontsize=8)
fig.tight_layout()
fig

# %% [markdown]
# ### Mean β per failure mode and load (eq. 8.12)
#
# CodeCal 2.0 against the table of the AHG document — identical to the
# two decimals printed there.

# %%
tab = cc.beta_table(sweeps['8.12'])[['wind', 'snow', 'imposed']]
comparison = pd.concat({'CodeCal 2.0': tab.round(2), 'AHG 2021': AHG_TABLE_812,
                        'difference': (tab - AHG_TABLE_812).round(2)}, axis=1)
comparison

# %% [markdown]
# ## 2. Calibration to $\beta_t = 3.8$
#
# Four calibrations, as in the study: one common $\gamma_Q$ or three
# load-specific ones, for the single-combination rule (8.12) and for the
# two-combination rule (8.13) with ξ free. The target is the Eurocode
# target for a 50-year reference period. Each takes 20–60 evaluations of
# the 810 situations.

# %%
runs = {
    'cal 8.12, common γ_Q': dict(free=('gamma_G', 'gamma_Q'), equation='8.12'),
    'cal 8.12, per load':   dict(free=('gamma_G', 'gamma_Q:wind', 'gamma_Q:snow', 'gamma_Q:imposed'), equation='8.12'),
    'cal 8.13, common γ_Q': dict(free=('xi', 'gamma_G', 'gamma_Q'), equation='8.13'),
    'cal 8.13, per load':   dict(free=('xi', 'gamma_G', 'gamma_Q:wind', 'gamma_Q:snow', 'gamma_Q:imposed'), equation='8.13'),
}
results = {}
for name, kw in runs.items():
    t0 = time.time()
    results[name] = cc.calibrate(model, beta_target=3.8, verbose=False, **kw)
    print(f'{name}: {len(results[name].history)} evaluations, {time.time() - t0:.0f} s')
    print('   ', str(results[name]).replace('\n', '\n    '))

# %%
rows = []
for name, res in results.items():
    ref = AHG[name]
    for p in res.free:
        rows.append({'calibration': name, 'parameter': p, 'CodeCal 2.0': round(res.factors.get(p), 3),
                     'AHG 2021': ref.get(p), 'difference': round(res.factors.get(p) - ref.get(p), 3)})
    rows.append({'calibration': name, 'parameter': 'weighted-mean β', 'CodeCal 2.0': round(res.beta_after, 3),
                 'AHG 2021': ref['beta'], 'difference': round(res.beta_after - ref['beta'], 3)})
pd.DataFrame(rows).set_index(['calibration', 'parameter'])

# %% [markdown]
# The calibrated factors agree with the study to ±0.005 — the resolution
# of the optimiser (the one larger deviation, the weighted-mean β of the
# per-load (8.13) calibration, 3.82 against 3.86, is the only number of
# the document that is not reproduced; the factors themselves are):
# $\gamma_G$ comes down to about 1.2, $\gamma_Q$ goes up to 1.75 with one
# common factor; with load-specific factors snow gets the largest one
# (2.35) and the imposed load the smallest (1.60), because their
# characteristic values sit at very different fractiles of the
# 50-year maximum of Θ_Q·Q (see {doc}`model`). The weighted-mean β is
# 3.8 in every case — the target is met on average by construction; what
# the extra parameters buy is a smaller spread.

# %% [markdown]
# ### The spread of β before and after calibration (eq. 8.12)
#
# The standard deviation of β over the design situations drops from
# about 0.46 to about 0.24 with load-specific factors; the remaining
# spread is between the materials, not within them.

# %%
spread = pd.DataFrame({
    'status quo': stats(sweeps['8.12']),
    'calibrated, common γ_Q': stats(results['cal 8.12, common γ_Q'].sweep_after),
    'calibrated, per load': stats(results['cal 8.12, per load'].sweep_after),
}).T
spread['AHG std'] = [AHG['std 8.12']['status quo'], None, AHG['std 8.12']['calibrated per load']]
spread

# %%
res = results['cal 8.12, per load']
fig = plots.plot_beta_distribution(res.sweep_after, 'mode', 3.8, ref_df=res.sweep_before,
                                   labels=('calibrated (per load)', 'present factors'),
                                   title='Eq. 8.12: present factors against load-specific calibrated factors')
fig

# %%
fig = plots.plot_beta_vs_aQ(res.sweep_after, 3.8, ref_df=res.sweep_before,
                            title='β against a_Q — calibrated (blue) and present factors (red)')
fig

# %%
fig = plots.plot_gamma_comparison(results, title='Calibrated partial factors of the four calibrations')
fig

# %% [markdown]
# ## 3. What the study concluded
#
# In the words of {cite}`KohlerChapter2026`, condensed:
#
# * Which combination rule a country adopts matters: (8.12) gives the
#   highest average reliability (3.74, closest to the target of 3.8),
#   (8.13) and (8.14) are 0.3 lower with the recommended ξ = 0.85. All absolute values are
#   conditional on the modelling assumptions; twenty years of use of all
#   three rules without evidence of unacceptable failure rates say that
#   the structural performance is acceptable with regard to public safety.
# * The more robust finding is the *spread*: with (8.12), β ranges from
#   2.4 to 4.5 (standard deviation 0.46). That is the price of a very
#   simple format — the same $\gamma_Q$ for wind, snow and imposed load
#   although their model uncertainties differ — and it means a
#   sub-optimal use of material in a sector that matters for
#   sustainability.
# * A moderate increase in complexity — load-specific $\gamma_Q$ —
#   halves the standard deviation (0.46 → 0.24). Calibrating the material
#   factors as well would reduce it further.
# * The committee's position during the revision was that there was
#   "no appetite for change" of the partial factors; the study documents
#   the potential for a later recalibration.
#
# The lesson notebooks ({doc}`../../lessons/notebooks/05_calibration`) take
# the calibration apart step by step; the {doc}`../manual/lab_mode` lets
# you repeat any of the runs above with your own model.

# %%
summary = {name: {**{p: round(r.factors.get(p), 3) for p in r.free}, 'beta': round(r.beta_after, 3),
                  'objective': round(r.objective, 4), 'evaluations': len(r.history)} for name, r in results.items()}
summary['status quo 8.12'] = {'beta': round(cc.weighted_mean_beta(sweeps['8.12']), 3),
                              'objective': round(cc.objective(sweeps['8.12'], 3.8), 4)}
import json
with open('results_summary.json', 'w') as f:
    json.dump(summary, f, indent=1)
print(json.dumps(summary, indent=1))
