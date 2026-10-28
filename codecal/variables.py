"""
Random variables of the calibration model and their characteristic values.

A `RandomVariable` bundles a name, a distribution and the *characteristic
value* x_k that the design equation uses for it. All variables of the AHG
model are normalised: the mean of a resistance, a model uncertainty, a
self-weight or a 50-year load maximum is 1.0, and the characteristic value
is expressed in the same units (e.g. x_k = 0.83 for steel strength means
"the characteristic value is 83 % of the mean"). Partial factors therefore
act on ratios, which is exactly what a calibration is about.

Characteristic values come from three sources:

    * a fractile of the distribution (5 % for strengths, 98 % of the yearly
      maximum for climatic loads, 50 % = mean for permanent loads);
    * a value pinned by the modeller (steel 0.83, model uncertainties 1.0);
    * for time-variant loads, a Monte-Carlo fractile of the *product* of
      the load and its model uncertainty — see `time_variant_load`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .distributions import Gumbel, Lognormal, Normal, distribution

Distribution = Normal | Lognormal | Gumbel


@dataclass(frozen=True)
class RandomVariable:
    """A basic variable with its characteristic value."""
    name: str
    dist: Distribution
    char: float
    symbol: str = ''
    note: str = ''

    # convenient pass-throughs ------------------------------------------
    @property
    def mean(self):
        return self.dist.mean

    @property
    def cov(self):
        return self.dist.cov

    def x_of_u(self, u):
        return self.dist.x_of_u(u)

    def dx_du(self, u):
        return self.dist.dx_du(u)

    def __str__(self):
        fam = type(self.dist).__name__
        return f"{self.name}: {fam}(mean={self.mean:.3g}, CoV={self.cov:.3g}), x_k={self.char:.3g}"


def rv(name, family, mean, cov, *, fractile=None, char=None, symbol='', note=''):
    """Build a RandomVariable from (family, mean, CoV) and either a fractile
    or a pinned characteristic value.

    >>> steel = rv('Steel yield strength', 'lognormal', 1.0, 0.05, char=0.83)
    >>> concrete = rv('Concrete strength', 'lognormal', 1.0, 0.10, fractile=0.05)
    """
    dist = distribution(family, mean, cov)
    if (fractile is None) == (char is None):
        raise ValueError(f"{name}: give exactly one of fractile= or char=.")
    if char is None:
        char = float(dist.ppf(fractile))
        note = note or f"{fractile:.0%} fractile"
    else:
        note = note or "pinned"
    return RandomVariable(name, dist, float(char), symbol, note)


def characteristic_value_mcs(theta_Q: RandomVariable, Q_basic: Gumbel, fractile,
                             n=10_000_000, rng=None):
    """Fractile of the product Θ_Q · Q by Monte-Carlo simulation.

    The characteristic value of a climatic load in the Eurocodes is a
    fractile of the yearly maximum (98 %), but the structure sees the load
    *and* the model uncertainty of its effect; the AHG defines the
    characteristic value on the product. There is no closed form for the
    fractile of a lognormal × Gumbel product, hence the simulation.

    Draw order (lognormal first, then uniform → Gumbel) and the sorted-index
    fractile follow the 2021 implementation so that seeded runs reproduce
    the published numbers.
    """
    rng = np.random if rng is None else rng
    lam, zeta = theta_Q.dist.lam, theta_Q.dist.zeta
    theta = rng.lognormal(lam, zeta, int(n))
    q = Q_basic.b - np.log(-np.log(rng.rand(int(n)))) / Q_basic.a
    product = np.sort(theta * q)
    return float(product[int(fractile * int(n))])


def time_variant_load(name, mean, cov, T_input, T_basic, T_ref, theta_Q,
                      fractile, *, char=None, n_mcs=10_000_000, rng=None,
                      symbol='', note=''):
    """Build the Gumbel variable of a time-variant load for the analysis.

    Parameters
    ----------
    mean, cov : moments of the maximum over the period `T_input` (years);
                the AHG gives the 50-year maximum with mean 1.
    T_basic   : basic recurrence period of the load (1 a for wind and snow,
                5 a for imposed loads). A reference period shorter than
                T_basic makes no sense for a maximum model, so the analysis
                period is T_an = max(T_ref, T_basic).
    theta_Q   : model uncertainty of the load effect (lognormal RV).
    fractile  : fractile over T_basic defining the characteristic value.
    char      : pin the characteristic value instead of simulating it.

    Returns the RandomVariable for the analysis period — mean normalised to
    1.0, same σ as the input (the Gumbel period shift leaves σ unchanged,
    so the CoV carries the whole effect of the reference period) — and
    the basic-period distribution used for the characteristic value.
    """
    T_an = max(float(T_ref), float(T_basic))
    g_input = Gumbel.from_moments(mean, cov)
    g_an = g_input.over_period(T_an / T_input)          # maximum over T_an
    scale = 1.0 / g_an.mean                             # normalise: mean(T_an) = 1
    g_an = Gumbel.from_moments(1.0, g_an.std * scale)   # same σ·scale, mean 1
    g_basic = g_an.over_period(T_basic / T_an)          # maximum over T_basic
    if char is None:
        char = characteristic_value_mcs(theta_Q, g_basic, fractile, n_mcs, rng)
        note = note or f"{fractile:.0%} fractile of Θ_Q·Q over {T_basic:g} a (MCS)"
    else:
        note = note or "pinned"
    load = RandomVariable(name, g_an, float(char), symbol, note)
    return load, g_basic
