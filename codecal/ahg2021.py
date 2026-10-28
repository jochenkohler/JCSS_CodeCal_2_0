"""
The probabilistic model of the AHG "Reliability Background of the
Eurocodes" calibration (input of November 2021, T. Vrouwenvelder et al.).

Every number below is data, not code: change it here (or edit the
dictionary `ahg_spec()` returns, or the JSON the lab GUI saves), nothing
else. All means are 1.0 (normalised variables), so the model is a table of
coefficients of variation, characteristic values and weights.

Conventions
    * Resistance model uncertainties Θ_R may carry a bias (mean ≠ 1),
      their characteristic value is 1.0 by definition.
    * Material strengths: 5 % fractile, except steel and aluminium where
      the 2021 input pinned 0.83 / 0.85 ("fractile chosen such that the
      characteristic value complies with the value adopted in June 2021;
      was this value adopted in the AHG calculations?" — still open; the
      5 % fractile with CoV 0.05 would be 0.92).
    * Self-weights: normal, characteristic value = mean or a pinned value.
    * Time-variant loads: moments of the 50-year maximum (mean 1), basic
      period 1 a (wind, snow) or 5 a (imposed); the characteristic value
      is the 98 % (99 % for imposed) fractile over the basic period of the
      product Θ_Q·Q, by Monte Carlo; the imposed load's was pinned at 1.35.
    * Buckling modes of the 2021 input, not part of the default set:
        steel buckling   Θ_R (1.0, 0.10), a_Q (0.2, 0.8), weight 0
        glulam buckling  Θ_R (1.0, 0.15), a_Q (0.3, 0.8), weight 0.02
        timber buckling  Θ_R (1.0, 0.15), a_Q (0.3, 0.8), weight 0.01
        alu buckling     Θ_R (1.0, 0.10), a_Q (0.4, 0.8), weight 0.01
"""

from __future__ import annotations

import copy

from .model import CalibrationModel, PartialFactors
from .spec import build_model

DEFAULT_SEED = 12345        # seeds the Monte-Carlo characteristic values
DEFAULT_N_MCS = 10_000_000

START_VALUES = PartialFactors(gamma_G=1.35, gamma_GP=1.35,
                              gamma_Q={'wind': 1.5, 'snow': 1.5, 'imposed': 1.5},
                              xi=0.85)
PSI_0 = {'wind': 0.6, 'snow': 0.5, 'imposed': 0.7}


def _rv(name, family, mean, cov, *, char=None, fractile=None, note=''):
    d = {'name': name, 'family': family, 'mean': mean, 'cov': cov}
    if char is not None:
        d['char'] = char
    else:
        d['fractile'] = fractile
    if note:
        d['note'] = note
    return d


def ahg_spec(T_ref=50, n_aQ=10, n_aG=3, beta_target=3.8, n_mcs=DEFAULT_N_MCS, seed=DEFAULT_SEED) -> dict:
    """The AHG model as a specification dictionary (see codecal.spec)."""
    BY_DEF = 'by definition'

    # -- self-weights of the materials (normal) ------------------------------
    G_steel    = _rv('Self-weight steel',     'normal', 1.0, 0.025, char=1.0)
    G_concrete = _rv('Self-weight concrete',  'normal', 1.0, 0.05,  char=0.98)
    G_glulam   = _rv('Self-weight glulam',    'normal', 1.0, 0.10,  char=0.95)
    G_timber   = _rv('Self-weight timber',    'normal', 1.0, 0.10,  char=0.95)
    G_masonry  = _rv('Self-weight masonry',   'normal', 1.0, 0.07,  fractile=0.5)
    G_alu      = _rv('Self-weight aluminium', 'normal', 1.0, 0.04,  fractile=0.5)
    G_soil     = _rv('Self-weight soil',      'normal', 1.0, 0.05,  fractile=0.5)

    # -- material strengths (lognormal) ---------------------------------------
    R_steel    = _rv('Steel yield strength',          'lognormal', 1.0, 0.05,  char=0.83, note='pinned (2021 input)')
    R_concrete = _rv('Concrete compressive strength', 'lognormal', 1.0, 0.10,  fractile=0.05)
    R_rebar    = _rv('Rebar yield strength',          'lognormal', 1.0, 0.045, fractile=0.05)
    R_glulam   = _rv('Glulam bending strength',       'lognormal', 1.0, 0.15,  fractile=0.05)
    R_timber   = _rv('Timber strength',               'lognormal', 1.0, 0.20,  fractile=0.05)
    R_masonry  = _rv('Masonry strength',              'lognormal', 1.0, 0.16,  fractile=0.05)
    R_alu      = _rv('Aluminium strength',            'lognormal', 1.0, 0.05,  char=0.85, note='pinned (2021 input)')
    R_soil     = _rv('Soil strength',                 'lognormal', 1.0, 0.15,  fractile=0.05)
    R_cpt      = _rv('CPT-based resistance',          'lognormal', 1.0, 0.12,  fractile=0.05)

    def mode(name, tR_name, tR_mean, tR_cov, R, gamma_M, G, a_Q, a_G, weight):
        return {'name': name,
                'theta_R': _rv(tR_name, 'lognormal', tR_mean, tR_cov, char=1.0, note=BY_DEF),
                'R': copy.deepcopy(R), 'gamma_M': gamma_M, 'G': copy.deepcopy(G),
                'a_Q': list(a_Q), 'a_G': list(a_G), 'weight': weight, 'include': True}

    # -- failure modes: Θ_R (mean, CoV), R, γ_M, G, a_Q range, a_G range, weight
    modes = [
        mode('Steel bending',        'Θ_R steel bending',     1.15, 0.05, R_steel,    1.00, G_steel,    (0.3, 0.8), (0.6, 1.0), 0.17),
        mode('Concrete compression', 'Θ_R concrete comp.',    0.97, 0.14, R_concrete, 1.50, G_concrete, (0.1, 0.7), (0.6, 1.0), 0.12),
        mode('Concrete bending',     'Θ_R concrete bend.',    1.03, 0.07, R_rebar,    1.15, G_concrete, (0.1, 0.7), (0.6, 1.0), 0.40),
        mode('Glulam bending',       'Θ_R glulam bending',    1.00, 0.10, R_glulam,   1.25, G_glulam,   (0.2, 0.8), (0.6, 1.0), 0.035),
        mode('Timber bending',       'Θ_R timber bending',    1.00, 0.10, R_timber,   1.30, G_timber,   (0.2, 0.8), (0.6, 1.0), 0.017),
        mode('Masonry',              'Θ_R masonry',           1.16, 0.18, R_masonry,  1.50, G_masonry,  (0.1, 0.7), (0.6, 1.0), 0.12),
        mode('Aluminium bending',    'Θ_R aluminium bend.',   1.27, 0.15, R_alu,      1.10, G_alu,      (0.3, 0.8), (0.6, 1.0), 0.017),
        mode('Pile foundation',      'Θ_R pile foundation',   1.00, 0.20, R_cpt,      1.50, G_soil,     (0.1, 0.5), (0.6, 1.0), 0.06),
        mode('Shallow foundation',   'Θ_R shallow found.',    1.00, 0.15, R_soil,     1.40, G_soil,     (0.1, 0.5), (0.6, 1.0), 0.06),
    ]

    # -- loads: Θ_Q (mean, CoV); Q = 50-year maximum (mean 1, CoV); T_basic; fractile over T_basic; ψ₀; weight
    def load(name, tQ_name, tQ_mean, tQ_cov, cov, T_basic, fractile, char, psi_0, weight, note=''):
        return {'name': name, 'theta_Q': _rv(tQ_name, 'lognormal', tQ_mean, tQ_cov, char=1.0, note=BY_DEF),
                'mean': 1.0, 'cov': cov, 'T_input': 50, 'T_basic': T_basic,
                'fractile': fractile, 'char': char, 'psi_0': psi_0, 'weight': weight,
                'include': True, **({'note': note} if note else {})}

    loads = [
        load('wind',    'Wind model uncertainty',    0.97, 0.26, 0.14, 1, 0.98, None, PSI_0['wind'],    0.4),
        load('snow',    'Snow model uncertainty',    0.81, 0.26, 0.20, 1, 0.98, None, PSI_0['snow'],    0.1),
        load('imposed', 'Imposed model uncertainty', 1.00, 0.10, 0.26, 5, 0.99, 1.35, PSI_0['imposed'], 0.5,
             note='q_k pinned in the 2021 input (the MCS would give ≈ 1.38)'),
    ]

    return {
        'name': 'AHG 2021 — Reliability Background of the Eurocodes',
        'T_ref': T_ref, 'n_aQ': n_aQ, 'n_aG': n_aG, 'beta_target': beta_target,
        'seed': seed, 'n_mcs': n_mcs,
        'theta_E': _rv('Load-effect model uncertainty', 'lognormal', 1.0, 0.10, char=1.0, note=BY_DEF),
        'P': _rv('Permanent load (other than self-weight)', 'normal', 1.0, 0.10, fractile=0.5),
        'modes': modes, 'loads': loads,
    }


def ahg_model(T_ref=50, n_aQ=10, n_aG=3, beta_target=3.8,
              n_mcs=DEFAULT_N_MCS, seed=DEFAULT_SEED) -> CalibrationModel:
    """Build the AHG model. `seed=None` leaves numpy's global random state
    in charge (not reproducible)."""
    return build_model(ahg_spec(T_ref, n_aQ, n_aG, beta_target, n_mcs, seed))
