"""
Probability distributions used in code calibration.

Three families cover the whole AHG model: Normal (self-weights, permanent
loads), Lognormal (resistances, model uncertainties) and Gumbel (yearly or
50-year maxima of wind, snow and imposed loads).

Every distribution is created from its **mean and coefficient of
variation** — the way probabilistic models are tabulated in the JCSS
Probabilistic Model Code — and offers the four operations the reliability
methods need:

    ppf(p)      fractile (quantile), e.g. the 5 % fractile of a strength
    cdf(x)      distribution function
    x_of_u(u)   the iso-probabilistic transform  x = F⁻¹(Φ(u))
    dx_du(u)    its derivative (chain rule for the gradient in u-space)

FORM works in the space of independent standard-normal variables u; the
transform x_of_u maps a point of that space back to the physical variable,
one variable at a time (the variables of the calibration model are
independent, so no Nataf/Rosenblatt transformation is needed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

EULER_GAMMA = 0.5772156649015329          # Euler–Mascheroni constant
GUMBEL_SHIFT = np.sqrt(6.0) / np.pi       # (√6/π) — appears in all Gumbel formulas


def _log_phi(u):
    """ln Φ(u), accurate also for large u where Φ(u) rounds to 1."""
    return special.log_ndtr(u)


@dataclass(frozen=True)
class Normal:
    """Normal distribution N(mean, std)."""
    mean: float
    std: float

    @classmethod
    def from_moments(cls, mean, cov):
        return cls(float(mean), float(mean) * float(cov))

    @property
    def cov(self):
        return self.std / self.mean

    def ppf(self, p):
        return self.mean + self.std * special.ndtri(p)

    def cdf(self, x):
        return special.ndtr((x - self.mean) / self.std)

    def pdf(self, x):
        return stats.norm.pdf(x, self.mean, self.std)

    def x_of_u(self, u):
        return self.mean + self.std * u

    def dx_du(self, u):
        return self.std * np.ones_like(np.asarray(u, dtype=float))


@dataclass(frozen=True)
class Lognormal:
    """Lognormal distribution: ln X ~ N(lam, zeta).

    From mean μ and CoV V:  zeta² = ln(1 + V²),  lam = ln μ − zeta²/2.
    """
    lam: float
    zeta: float

    @classmethod
    def from_moments(cls, mean, cov):
        if mean <= 0:
            raise ValueError("A lognormal variable needs a positive mean.")
        zeta = np.sqrt(np.log(1.0 + cov ** 2))
        lam = np.log(mean) - 0.5 * zeta ** 2
        return cls(float(lam), float(zeta))

    @property
    def mean(self):
        return np.exp(self.lam + 0.5 * self.zeta ** 2)

    @property
    def std(self):
        return self.mean * self.cov

    @property
    def cov(self):
        return np.sqrt(np.exp(self.zeta ** 2) - 1.0)

    def ppf(self, p):
        return np.exp(self.lam + self.zeta * special.ndtri(p))

    def cdf(self, x):
        return special.ndtr((np.log(x) - self.lam) / self.zeta)

    def pdf(self, x):
        return stats.lognorm.pdf(x, self.zeta, scale=np.exp(self.lam))

    def x_of_u(self, u):
        return np.exp(self.lam + self.zeta * u)

    def dx_du(self, u):
        return self.zeta * np.exp(self.lam + self.zeta * u)


@dataclass(frozen=True)
class Gumbel:
    """Gumbel (type I largest) distribution  F(x) = exp(−exp(−a (x − b))).

    Mean μ = b + γ/a,  standard deviation σ = π/(a √6).

    The distribution of the maximum over T₂ periods of a Gumbel variable
    that describes the maximum over one period T₁ is again Gumbel with the
    same a (same σ) and b shifted by ln(T₂/T₁)/a — see `over_period`. This
    is how a yearly-maximum wind load becomes the 50-year maximum.
    """
    a: float
    b: float

    @classmethod
    def from_moments(cls, mean, cov):
        std = mean * cov
        a = np.pi / (np.sqrt(6.0) * std)
        b = mean - EULER_GAMMA / a
        return cls(float(a), float(b))

    @property
    def mean(self):
        return self.b + EULER_GAMMA / self.a

    @property
    def std(self):
        return np.pi / (np.sqrt(6.0) * self.a)

    @property
    def cov(self):
        return self.std / self.mean

    def over_period(self, factor):
        """Distribution of the maximum over `factor` times the current
        period (factor = T_new / T_current; may be < 1 to go back)."""
        return Gumbel(self.a, self.b + np.log(factor) / self.a)

    def ppf(self, p):
        return self.b - np.log(-np.log(p)) / self.a

    def cdf(self, x):
        return np.exp(-np.exp(-self.a * (x - self.b)))

    def pdf(self, x):
        y = -self.a * (x - self.b)
        return self.a * np.exp(y - np.exp(y))

    def x_of_u(self, u):
        # F⁻¹(Φ(u)) = b − ln(−ln Φ(u))/a, with ln Φ(u) from log_ndtr
        return self.b - np.log(-_log_phi(u)) / self.a

    def dx_du(self, u):
        # d/du of the above: −φ(u) / (a Φ(u) ln Φ(u))
        log_phi = _log_phi(u)
        return -np.exp(stats.norm.logpdf(u) - log_phi) / (self.a * log_phi)


FAMILIES = {'normal': Normal, 'lognormal': Lognormal, 'gumbel': Gumbel}


def distribution(family: str, mean: float, cov: float):
    """Create a distribution by family name from mean and CoV."""
    try:
        return FAMILIES[family.lower()].from_moments(mean, cov)
    except KeyError:
        raise ValueError(f"Unknown distribution family {family!r}; "
                         f"known: {sorted(FAMILIES)}") from None
