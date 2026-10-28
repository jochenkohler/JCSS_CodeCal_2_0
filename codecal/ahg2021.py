"""
The probabilistic model of the AHG "Reliability Background of the
Eurocodes" calibration (input of November 2021, T. Vrouwenvelder et al.).

Every number below is data, not code: change it here, nothing else. All
means are 1.0 (normalised variables), so the model is a table of
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
"""

from __future__ import annotations

import numpy as np

from .model import CalibrationModel, FailureMode, PartialFactors, VariableLoad
from .variables import rv, time_variant_load

DEFAULT_SEED = 12345        # seeds the Monte-Carlo characteristic values

START_VALUES = PartialFactors(gamma_G=1.35, gamma_GP=1.35,
                              gamma_Q={'wind': 1.5, 'snow': 1.5, 'imposed': 1.5},
                              xi=0.85)
PSI_0 = {'wind': 0.6, 'snow': 0.5, 'imposed': 0.7}


def ahg_model(T_ref=50, n_aQ=10, n_aG=3, beta_target=3.8,
              n_mcs=10_000_000, seed=DEFAULT_SEED) -> CalibrationModel:
    """Build the AHG model. `seed=None` leaves numpy's global random state
    in charge (not reproducible)."""

    # -- self-weights of the materials (normal) ------------------------------
    G_steel    = rv('Self-weight steel',     'normal', 1.0, 0.025, char=1.0,  symbol='G_steel')
    G_concrete = rv('Self-weight concrete',  'normal', 1.0, 0.05,  char=0.98, symbol='G_concrete')
    G_glulam   = rv('Self-weight glulam',    'normal', 1.0, 0.10,  char=0.95, symbol='G_glulam')
    G_timber   = rv('Self-weight timber',    'normal', 1.0, 0.10,  char=0.95, symbol='G_timber')
    G_masonry  = rv('Self-weight masonry',   'normal', 1.0, 0.07,  fractile=0.5, symbol='G_masonry')
    G_alu      = rv('Self-weight aluminium', 'normal', 1.0, 0.04,  fractile=0.5, symbol='G_alu')
    G_soil     = rv('Self-weight soil',      'normal', 1.0, 0.05,  fractile=0.5, symbol='G_soil')

    # -- material strengths (lognormal) ---------------------------------------
    R_steel    = rv('Steel yield strength',          'lognormal', 1.0, 0.05,  char=0.83, symbol='R_steel')
    R_concrete = rv('Concrete compressive strength', 'lognormal', 1.0, 0.10,  fractile=0.05, symbol='R_concrete')
    R_rebar    = rv('Rebar yield strength',          'lognormal', 1.0, 0.045, fractile=0.05, symbol='R_rebar')
    R_glulam   = rv('Glulam bending strength',       'lognormal', 1.0, 0.15,  fractile=0.05, symbol='R_glulam')
    R_timber   = rv('Timber strength',               'lognormal', 1.0, 0.20,  fractile=0.05, symbol='R_timber')
    R_masonry  = rv('Masonry strength',              'lognormal', 1.0, 0.16,  fractile=0.05, symbol='R_masonry')
    R_alu      = rv('Aluminium strength',            'lognormal', 1.0, 0.05,  char=0.85, symbol='R_alu')
    R_soil     = rv('Soil strength',                 'lognormal', 1.0, 0.15,  fractile=0.05, symbol='R_soil')
    R_cpt      = rv('CPT-based resistance',          'lognormal', 1.0, 0.12,  fractile=0.05, symbol='R_CPT')

    def theta_R(name, mean, cov):
        return rv(name, 'lognormal', mean, cov, char=1.0, symbol='θ_R', note='by definition')

    # -- failure modes: (Θ_R mean, CoV), R, γ_M, G, a_Q range, a_G range, weight
    modes = [
        FailureMode('Steel bending',        theta_R('Θ_R steel bending', 1.15, 0.05), R_steel,    1.00, G_steel,    (0.3, 0.8), (0.6, 1.0), 0.17),
        FailureMode('Concrete compression', theta_R('Θ_R concrete comp.', 0.97, 0.14), R_concrete, 1.50, G_concrete, (0.1, 0.7), (0.6, 1.0), 0.12),
        FailureMode('Concrete bending',     theta_R('Θ_R concrete bend.', 1.03, 0.07), R_rebar,    1.15, G_concrete, (0.1, 0.7), (0.6, 1.0), 0.40),
        FailureMode('Glulam bending',       theta_R('Θ_R glulam bending', 1.00, 0.10), R_glulam,   1.25, G_glulam,   (0.2, 0.8), (0.6, 1.0), 0.035),
        FailureMode('Timber bending',       theta_R('Θ_R timber bending', 1.00, 0.10), R_timber,   1.30, G_timber,   (0.2, 0.8), (0.6, 1.0), 0.017),
        FailureMode('Masonry',              theta_R('Θ_R masonry',        1.16, 0.18), R_masonry,  1.50, G_masonry,  (0.1, 0.7), (0.6, 1.0), 0.12),
        FailureMode('Aluminium bending',    theta_R('Θ_R aluminium bend.', 1.27, 0.15), R_alu,      1.10, G_alu,      (0.3, 0.8), (0.6, 1.0), 0.017),
        FailureMode('Pile foundation',      theta_R('Θ_R pile foundation', 1.00, 0.20), R_cpt,      1.50, G_soil,     (0.1, 0.5), (0.6, 1.0), 0.06),
        FailureMode('Shallow foundation',   theta_R('Θ_R shallow found.',  1.00, 0.15), R_soil,     1.40, G_soil,     (0.1, 0.5), (0.6, 1.0), 0.06),
    ]
    # Buckling modes of the 2021 input, not part of the default set:
    #   steel buckling   Θ_R (1.0, 0.10), a_Q (0.2, 0.8), weight 0
    #   glulam buckling  Θ_R (1.0, 0.15), a_Q (0.3, 0.8), weight 0.02
    #   timber buckling  Θ_R (1.0, 0.15), a_Q (0.3, 0.8), weight 0.01
    #   alu buckling     Θ_R (1.0, 0.10), a_Q (0.4, 0.8), weight 0.01

    # -- load side ---------------------------------------------------------------
    theta_E = rv('Load-effect model uncertainty', 'lognormal', 1.0, 0.10, char=1.0, symbol='θ_E', note='by definition')
    P = rv('Permanent load (other than self-weight)', 'normal', 1.0, 0.10, fractile=0.5, symbol='P')

    theta_wind    = rv('Wind model uncertainty',    'lognormal', 0.97, 0.26, char=1.0, symbol='θ_w', note='by definition')
    theta_snow    = rv('Snow model uncertainty',    'lognormal', 0.81, 0.26, char=1.0, symbol='θ_s', note='by definition')
    theta_imposed = rv('Imposed model uncertainty', 'lognormal', 1.00, 0.10, char=1.0, symbol='θ_i', note='by definition')

    rng = np.random if seed is None else np.random.RandomState(seed)
    # (mean, CoV) of the 50-year maximum; T_basic; fractile over T_basic
    Q_wind, _ = time_variant_load('Wind, 50-year maximum', 1.0, 0.14, T_input=50, T_basic=1,
                                  T_ref=T_ref, theta_Q=theta_wind, fractile=0.98,
                                  n_mcs=n_mcs, rng=rng, symbol='Q_w')
    Q_snow, _ = time_variant_load('Snow, 50-year maximum', 1.0, 0.20, T_input=50, T_basic=1,
                                  T_ref=T_ref, theta_Q=theta_snow, fractile=0.98,
                                  n_mcs=n_mcs, rng=rng, symbol='Q_s')
    Q_imposed, _ = time_variant_load('Imposed, 50-year maximum', 1.0, 0.26, T_input=50, T_basic=5,
                                     T_ref=T_ref, theta_Q=theta_imposed, fractile=0.99,
                                     char=1.35, symbol='Q_i',
                                     note='pinned in the 2021 input (the MCS would give ≈ 1.38)')

    loads = [
        VariableLoad('wind',    theta_wind,    Q_wind,    PSI_0['wind'],    0.4, T_basic=1),
        VariableLoad('snow',    theta_snow,    Q_snow,    PSI_0['snow'],    0.1, T_basic=1),
        VariableLoad('imposed', theta_imposed, Q_imposed, PSI_0['imposed'], 0.5, T_basic=5),
    ]

    return CalibrationModel(
        failure_modes=modes, loads=loads, theta_E=theta_E, P=P,
        T_ref=T_ref, n_aQ=n_aQ, n_aG=n_aG, beta_target=beta_target,
        description='AHG "Reliability Background of the Eurocodes", input of November 2021',
    )
