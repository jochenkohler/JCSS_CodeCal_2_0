# %% [markdown]
# # The probabilistic model
#
# Step 4 of the procedure: the failure modes and the probabilistic models
# of the variables, as agreed in the Ad-hoc Group (input of November
# 2021, {cite}`KoehlerHingoraniSorensen2021`). The tables on this page are
# generated from `codecal.ahg_spec()` — the model as the code holds it — so
# they cannot drift from what the tool computes. The same numbers appear
# in the *Model* tab of the lab mode.
#
# **Conventions.** All variables are normalised to mean 1 (except the
# resistance and load model uncertainties, whose mean expresses a bias),
# so the model is a table of coefficients of variation, characteristic
# values and weights:
#
# * Resistance model uncertainties $\Theta_R$ are lognormal; their
#   characteristic value is 1 by definition. A mean of 1.15 means that the
#   real capacity is, on average, 15 % larger than the model predicts.
# * Material strengths are lognormal with the 5 % fractile as
#   characteristic value — except steel and aluminium, whose
#   characteristic values were assigned by the experts of the group
#   (0.83 and 0.85, corresponding to the 0.01 % and 0.06 % fractiles),
#   following the nominal values of EN 1993-1-1 Annex E
#   {cite}`EN1993_1_1`.
# * Self-weights are normal with the mean (or a pinned value slightly
#   below it) as characteristic value; the other permanent load $G_P$
#   (called $P$ in the code) is normal with CoV 0.10 and the mean as
#   characteristic value.
# * Variable loads are represented by the 50-year maximum (Gumbel, mean 1)
#   with a basic period of 1 year (wind, snow) or 5 years (imposed). The
#   characteristic value is the 98 % fractile (99 % for the imposed load)
#   over the basic period of the *product* $\Theta_Q \cdot Q$, obtained by
#   Monte-Carlo simulation. For the imposed load the simulation would give
#   1.38; the value 1.35 was conventionally assigned in the study.
# * $\Theta_E$, the model uncertainty of the load effect, is lognormal
#   with CoV 0.10 and characteristic value 1.
# * The load proportions $a_Q$ and $a_G$ are varied uniformly between the
#   limits per failure mode (10 values of $a_Q$, 3 of $a_G$); the weight of
#   a design situation is $w_{\text{mode}} \, w_{\text{load}} / (n_{a_G} n_{a_Q})$.
#   The weights are expert judgement of how frequent the situations are;
#   the results are moderately sensitive to them.

# %% tags=["hide-input"]
import os
import sys

import pandas as pd

try:
    import codecal as cc
except ImportError:
    sys.path.insert(0, os.path.abspath('../..'))
    import codecal as cc

pd.set_option('display.width', 140)
spec = cc.ahg_spec()
model = cc.build_model({**spec, 'n_mcs': 2_000_000})     # 2 M draws for the page; the tool uses 10 M
print(spec['name'])


# %% [markdown]
# ## Failure modes: resistance model uncertainty, strength, γ_M

# %%
def xk_label(d):
    """'0.83 (pinned)' or '5% fractile' for a random-variable entry of the spec."""
    return f"{d['char']} (pinned)" if d.get('char') is not None else f"{d['fractile']:.0%} fractile"

rows = []
for m in spec['modes']:
    rows.append({'failure mode': m['name'],
                 'Θ_R mean': m['theta_R']['mean'], 'Θ_R CoV': m['theta_R']['cov'],
                 'R': m['R']['name'], 'R CoV': m['R']['cov'],
                 'R x_k': xk_label(m['R']),
                 'γ_M': m['gamma_M']})
pd.DataFrame(rows).set_index('failure mode')

# %% [markdown]
# The resulting characteristic values of the strengths (the pinned ones
# and the 5 % fractiles of the lognormal distributions):

# %%
pd.DataFrame([{'failure mode': m.name, 'R': m.R.name, 'family': type(m.R.dist).__name__, 'CoV': m.R.cov,
               'x_k': round(m.R.char, 3), 'note': m.R.note} for m in model.failure_modes]).set_index('failure mode')

# %% [markdown]
# ## Self-weight and permanent load

# %%
rows = [{'variable': m['G']['name'], 'used by': m['name'], 'family': m['G']['family'], 'CoV': m['G']['cov'],
         'x_k': xk_label(m['G'])} for m in spec['modes']]
rows.append({'variable': spec['P']['name'], 'used by': 'all', 'family': spec['P']['family'], 'CoV': spec['P']['cov'],
             'x_k': f"{spec['P']['fractile']:.0%} fractile"})
rows.append({'variable': spec['theta_E']['name'], 'used by': 'all', 'family': spec['theta_E']['family'],
             'CoV': spec['theta_E']['cov'], 'x_k': spec['theta_E']['char']})
pd.DataFrame(rows).set_index('variable')

# %% [markdown]
# ## Variable loads
#
# The CoV is that of the 50-year maximum; $T_{basic}$ is the period of
# independent re-occurrence; the characteristic value is the fractile of
# the basic-period maximum of $\Theta_Q \cdot Q$ (simulated, or pinned).

# %%
rows = []
for l in spec['loads']:
    rows.append({'load': l['name'], 'Θ_Q mean': l['theta_Q']['mean'], 'Θ_Q CoV': l['theta_Q']['cov'],
                 'Q CoV (50 a)': l['cov'], 'T_basic (a)': l['T_basic'], 'fractile': l['fractile'],
                 'q_k': 'simulated' if l.get('char') is None else f"pinned {l['char']}",
                 'ψ₀': l['psi_0'], 'weight': l['weight']})
pd.DataFrame(rows).set_index('load')

# %% [markdown]
# The simulated characteristic values (relative to the mean 50-year
# maximum) and what the model actually uses:

# %%
pd.DataFrame([{'load': l.name, 'q_k used': round(l.Q.char, 3), 'note': l.Q.note,
               'Q at T_an: CoV': round(l.Q.cov, 3)} for l in model.loads]).set_index('load')

# %% [markdown]
# ## Load proportions and weights per failure mode
#
# $a_Q$ and $a_G$ are varied uniformly between the limits; the weights of
# the failure modes sum to 1 (0.999 in the original input; CodeCal 2.0
# normalises the weights of the design situations to 1).

# %%
pd.DataFrame([{'failure mode': m['name'], 'a_Q min': m['a_Q'][0], 'a_Q max': m['a_Q'][1],
               'a_G min': m['a_G'][0], 'a_G max': m['a_G'][1], 'weight': m['weight']}
              for m in spec['modes']]).set_index('failure mode')

# %% [markdown]
# ## Global settings

# %%
pd.Series({'reference period T_ref (a)': spec['T_ref'], 'target β_t (50 a)': spec['beta_target'],
           'a_Q values per mode': spec['n_aQ'], 'a_G values per mode': spec['n_aG'],
           'design situations': 9 * 3 * spec['n_aQ'] * spec['n_aG'],
           'Monte-Carlo draws for q_k': spec['n_mcs'], 'seed': spec['seed']}).to_frame('value')

# %% [markdown]
# ## Modes not in the default set
#
# The 2021 input also listed buckling modes with weight zero or very
# small weight (steel, glulam, timber and aluminium buckling, with Θ_R CoV
# 0.10–0.15 and weights 0 / 0.02 / 0.01 / 0.01). They are not part of the
# default model; they can be added in the lab mode or in a JSON model file
# ({doc}`../manual/model_files`).
#
# The model as JSON — the file the lab mode saves and loads:

# %%
print(cc.to_json(spec)[:1500] + '\n...')
