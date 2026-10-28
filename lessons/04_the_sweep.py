# %% [markdown]
# # Lesson 4 — The sweep: how reliable are structures designed with the present Eurocode?
#
# A code is used for many structures. The AHG model represents them by
#
# * 9 failure modes (materials and mechanisms), each with a weight for how
#   common it is,
# * 3 dominating variable loads (wind, snow, imposed), also weighted,
# * a grid of load proportions: 3 values of $a_G$ (share of self-weight in
#   the permanent load) and 10 values of $a_Q$ (share of the variable load).
#
# That is 9 × 3 × 3 × 10 = 810 **design situations**, each with the weight
# $w = w_{mode}\,w_{load}/(n_{a_G} n_{a_Q})$. Evaluating all of them with
# the present partial factors is the *status quo* of the code — the
# starting point of every calibration.

# %%
import os
import sys
import time

import numpy as np
import pandas as pd

try:
    import codecal as cc
except ImportError:
    sys.path.insert(0, os.path.abspath('..'))
    import codecal as cc
from codecal import plots

FIG = 'figures'
os.makedirs(FIG, exist_ok=True)
pd.set_option('display.width', 120)


def save(fig, name):
    """Save a figure next to the lesson and show it (inline in Jupyter/Spyder)."""
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches='tight')
    try:
        from IPython.display import display
        display(fig)
    except ImportError:            # plain python: the PNG in figures/ is the output
        pass

# %% [markdown]
# ## 1. The model
#
# Everything is in `codecal/ahg2021.py` — a table of coefficients of
# variation, characteristic values and weights. Read it once; it is the
# whole "input" of the study.

# %%
model = cc.ahg_model()            # 10 M Monte-Carlo draws for the characteristic values (a few seconds)
print(model.summary())

# %% [markdown]
# ## 2. The status quo: β of every design situation with γ_G = 1.35, γ_Q = 1.5
#
# `evaluate` runs FORM for each situation and returns a table — one row per
# situation with its β, design parameter z and sensitivity factors.

# %%
t0 = time.time()
df = cc.evaluate(model, cc.START_VALUES, equation='8.12')
print(f"{len(df)} situations evaluated in {time.time() - t0:.1f} s")
print(df.head().round(3).to_string())

# %% [markdown]
# The headline number of the status quo is the **weighted-mean reliability
# index**. The AHG reports it as 3.74; with the weights normalised to sum
# to 1 it is 3.74 as well (the input weights sum to 0.999):

# %%
beta_mean = cc.weighted_mean_beta(df)
print(f"weighted-mean β = {beta_mean:.3f}   (target 3.8)")
print(f"range over the situations: {df['beta'].min():.2f} … {df['beta'].max():.2f}")

# %% [markdown]
# ## 3. Where is the reliability high, where is it low?
#
# Mean β per failure mode and load — the table the AHG discussed:

# %%
table = cc.beta_table(df)
print(table.round(2))

# %% [markdown]
# Snow-dominated structures are consistently below the target, imposed
# load above it; masonry and concrete compression are high (large
# resistance model uncertainty with bias, large γ_M), glulam and timber
# low. The spread within a group is the load ratio: the next figure shows
# the whole distribution, the one after it β against $a_Q$.

# %%
save(plots.plot_beta_distribution(df, by='mode', beta_target=3.8,
                                  title='Status quo: β per failure mode (all loads and load ratios)'),
     'lesson04_beta_by_mode.png')
save(plots.plot_beta_distribution(df, by='load', beta_target=3.8, title='Status quo: β per load'),
     'lesson04_beta_by_load.png')
save(plots.plot_beta_vs_aQ(df, beta_target=3.8, title='Status quo: β against the variable-load share'),
     'lesson04_beta_vs_aQ.png')

# %% [markdown]
# Every panel shows the same pattern: β falls with $a_Q$. With the present
# factors the code is more conservative for permanent-load-dominated
# structures than for variable-load-dominated ones. This is the
# "reliability differentiation" that a calibration of $\gamma_G$ versus
# $\gamma_Q$ tries to flatten.
#
# ## 4. Which uncertainties drive the result?
#
# The mean sensitivity factors $\alpha_i^2$ per failure mode say which
# variables the reliability depends on — and therefore which parts of the
# probabilistic model deserve the most scrutiny:

# %%
save(plots.plot_sensitivities(df, by='mode'), 'lesson04_sensitivities.png')
mean_alpha2 = (df.filter(like='alpha_') ** 2).mean().round(3)
print("mean α² over all situations:")
print(mean_alpha2.to_string())

# %% [markdown]
# The three model uncertainties — of the resistance ($\Theta_R$), of the
# load effect ($\Theta_E$) and of the variable load ($\Theta_Q$) — together
# carry more than half of the sensitivity; the self-weight and the
# permanent load almost nothing (their CoVs are small and their
# characteristic values are the means). The CoVs of the model
# uncertainties are also the least well-known numbers of the model.
# Lesson 3's exercise 3 (wind CoV 0.15 instead of 0.26) is worth doing now
# on the whole model.
#
# ## 5. Weights
#
# The weighted mean is dominated by concrete bending (w = 0.40) and the
# imposed load (w = 0.5). Try a different weighting and see how much the
# headline number moves — it is a modelling decision, not a measurement:

# %%
alt = df.copy()
alt['weight'] = 1.0 / len(alt)                                   # every situation equal
print(f"weighted-mean β, AHG weights : {beta_mean:.3f}")
print(f"weighted-mean β, equal weights: {cc.weighted_mean_beta(alt):.3f}")
df.to_csv('status_quo_sweep.csv', index=False)
print("sweep saved to status_quo_sweep.csv")

# %% [markdown]
# ## Exercises
#
# 1. Evaluate the status quo for eq. 8.13 and 8.14 (with ξ = 0.85). Which
#    one gives the lowest weighted-mean β, and why (see lesson 3, section 5)?
# 2. Set `n_aQ=20` and check that the weighted mean hardly changes — the grid
#    is fine enough.
# 3. Give the snow load the weight 0.4 and the wind load 0.1. How does the
#    weighted mean react, and which calibration result would you expect?
