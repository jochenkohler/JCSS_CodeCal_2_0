"""
The calibration problem: failure modes, loads, design situations.

A code calibration asks: *which partial factors make the reliability of
structures designed with the code closest to the target, over the range of
structures the code is used for?* Three things define that range here:

Failure modes  (`FailureMode`)
    one per material and mechanism — steel bending, concrete compression,
    glulam bending, ... Each has a resistance model uncertainty Θ_R, a
    material strength R with its partial factor γ_M, the self-weight G of
    that material, a range of load ratios and a weight (how common it is).

Variable loads  (`VariableLoad`)
    wind, snow, imposed — each with its model uncertainty Θ_Q, the
    time-variant load Q (Gumbel maximum over the reference period), the
    combination factor ψ₀ and a weight.

Load ratios
    a_Q = share of the variable load in the total characteristic load,
    a_G = share of the self-weight G in the permanent part (the rest is the
    other permanent load P). Every (mode, load, a_G, a_Q) is one
    *design situation* with weight w_mode · w_load / (n_aG · n_aQ).

The limit state of one design situation is

    g = z · Θ_R · R  −  Θ_E · [ (1 − a_Q)(a_G G + (1 − a_G) P) + a_Q Θ_Q Q ]

where z is the *design parameter*: the resistance the engineer provides
when designing with the code, obtained from the design equation with the
characteristic values and the partial factors. All variables have mean 1;
z therefore expresses how much mean resistance the code demands per unit
of mean load, and the reliability index of the situation follows from the
scatter of the seven variables around that ratio.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from itertools import product

import numpy as np
import pandas as pd

from .reliability import LimitState, form, importance_sampling, sorm
from .variables import RandomVariable


# ---- partial factors ----------------------------------------------------------

@dataclass(frozen=True)
class PartialFactors:
    """The reliability elements of the design equation.

    gamma_Q maps load name → γ_Q; a single value is used for all loads.
    xi is the reduction factor on permanent loads in eq. 8.13b / 8.14b.
    """
    gamma_G: float = 1.35          # self-weight (and, unless differentiated, all permanent load)
    gamma_GP: float = 1.35         # other permanent load
    gamma_Q: dict = field(default_factory=lambda: {'wind': 1.5, 'snow': 1.5, 'imposed': 1.5})
    xi: float = 0.85

    def gamma_Q_of(self, load_name):
        if isinstance(self.gamma_Q, dict):
            return self.gamma_Q[load_name]
        return float(self.gamma_Q)

    # -- free-parameter interface for the optimiser ---------------------
    # Parameter names: 'gamma_G', 'gamma_GP', 'xi', 'gamma_Q' (common to all
    # loads) or 'gamma_Q:wind' etc. (one per load).
    def get(self, name):
        if name.startswith('gamma_Q:'):
            return self.gamma_Q_of(name.split(':', 1)[1])
        if name == 'gamma_Q':
            values = set(self.gamma_Q.values()) if isinstance(self.gamma_Q, dict) else {self.gamma_Q}
            return float(values.pop()) if len(values) == 1 else float(np.mean(list(self.gamma_Q.values())))
        return getattr(self, name)

    def with_values(self, names, values):
        """A copy with the named parameters replaced by `values`.

        Conventions of the AHG degrees of freedom: setting 'gamma_Q' sets
        the factor of every load, and setting 'gamma_G' without 'gamma_GP'
        sets the common factor of both permanent loads (self-weight and
        other permanent load). Name 'gamma_GP' as well to differentiate.
        """
        names = list(names)
        kw = {}
        gq = dict(self.gamma_Q) if isinstance(self.gamma_Q, dict) else None
        for name, v in zip(names, values):
            v = float(v)
            if name == 'gamma_Q':
                gq = {k: v for k in gq} if gq is not None else v
            elif name.startswith('gamma_Q:'):
                gq[name.split(':', 1)[1]] = v
            elif name in ('gamma_G', 'gamma_GP', 'xi'):
                kw[name] = v
            else:
                raise ValueError(f"Unknown partial factor {name!r}")
        if 'gamma_G' in kw and 'gamma_GP' not in names:
            kw['gamma_GP'] = kw['gamma_G']
        if gq is not None:
            kw['gamma_Q'] = gq
        return replace(self, **kw)

    def as_dict(self):
        d = {'gamma_G': self.gamma_G, 'gamma_GP': self.gamma_GP, 'xi': self.xi}
        if isinstance(self.gamma_Q, dict):
            d.update({f'gamma_Q:{k}': v for k, v in self.gamma_Q.items()})
        else:
            d['gamma_Q'] = self.gamma_Q
        return d


# ---- model components ---------------------------------------------------------

@dataclass(frozen=True)
class FailureMode:
    name: str
    theta_R: RandomVariable        # resistance model uncertainty
    R: RandomVariable              # material strength (char = fractile or pinned)
    gamma_M: float                 # partial factor of the material (fixed by the material code)
    G: RandomVariable              # self-weight of the material
    a_Q_range: tuple               # (min, max) share of the variable load
    a_G_range: tuple               # (min, max) share of self-weight in the permanent load
    weight: float                  # relative frequency of this mode in practice

    def __str__(self):
        return (f"{self.name}: Θ_R mean {self.theta_R.mean:.2f} CoV {self.theta_R.cov:.2f}, "
                f"R CoV {self.R.cov:.3f} (x_k {self.R.char:.3f}), γ_M {self.gamma_M}, "
                f"a_Q {self.a_Q_range}, a_G {self.a_G_range}, w {self.weight}")


@dataclass(frozen=True)
class VariableLoad:
    name: str
    theta_Q: RandomVariable        # model uncertainty of the load effect
    Q: RandomVariable              # time-variant load, maximum over the analysis period
    psi_0: float                   # combination factor (eq. 8.13a)
    weight: float
    T_basic: float = 1.0

    def __str__(self):
        return (f"{self.name}: Θ_Q mean {self.theta_Q.mean:.2f} CoV {self.theta_Q.cov:.2f}, "
                f"Q Gumbel CoV {self.Q.cov:.3f} (q_k {self.Q.char:.3f}), ψ₀ {self.psi_0}, w {self.weight}")


@dataclass
class CalibrationModel:
    failure_modes: list
    loads: list
    theta_E: RandomVariable        # load-effect model uncertainty
    P: RandomVariable              # other permanent load
    T_ref: float = 50.0
    n_aQ: int = 10
    n_aG: int = 3
    beta_target: float = 3.8
    description: str = ''

    def design_situations(self):
        """All (mode, load, a_G, a_Q) combinations with their weights."""
        out = []
        for load, mode in product(self.loads, self.failure_modes):
            a_G_values = np.linspace(*mode.a_G_range, self.n_aG)
            a_Q_values = np.linspace(*mode.a_Q_range, self.n_aQ)
            w = load.weight * mode.weight / (self.n_aG * self.n_aQ)
            for a_G in a_G_values:
                for a_Q in a_Q_values:
                    out.append(DesignSituation(mode, load, float(a_G), float(a_Q), w, self))
        return out

    def total_weight(self):
        return sum(m.weight for m in self.failure_modes) * sum(l.weight for l in self.loads)

    def summary(self):
        lines = [self.description, '',
                 f"reference period {self.T_ref:g} a, target β {self.beta_target}",
                 f"{len(self.failure_modes)} failure modes × {len(self.loads)} loads × "
                 f"{self.n_aG} a_G × {self.n_aQ} a_Q = "
                 f"{len(self.failure_modes) * len(self.loads) * self.n_aG * self.n_aQ} design situations",
                 '', 'Failure modes:']
        lines += [f"  {m}" for m in self.failure_modes]
        lines += ['', 'Variable loads:'] + [f"  {l}" for l in self.loads]
        lines += ['', f"  {self.theta_E}", f"  {self.P}"]
        return '\n'.join(lines)


# ---- one design situation -----------------------------------------------------

EQUATIONS = ('8.12', '8.13', '8.14')


@dataclass
class DesignSituation:
    mode: FailureMode
    load: VariableLoad
    a_G: float
    a_Q: float
    weight: float
    model: CalibrationModel

    @property
    def variables(self):
        """Order of the seven variables in g: Θ_R, R, Θ_E, G, P, Θ_Q, Q."""
        return [self.mode.theta_R, self.mode.R, self.model.theta_E, self.mode.G,
                self.model.P, self.load.theta_Q, self.load.Q]

    @property
    def variable_names(self):
        return ['Θ_R', 'R', 'Θ_E', 'G', 'P', 'Θ_Q', 'Q']

    # -- design equation → design parameter z ---------------------------------
    def design_parameter(self, pf: PartialFactors, equation='8.12'):
        """z from the design equation, EN 1990:2023 clause 8.3.3.

        With characteristic values x_k and the load fractions a_G, a_Q, the
        design load effect is::

          8.12   E_d = (1−a_Q)[a_G γ_G g_k + (1−a_G) γ_GP p_k] + a_Q γ_Q q_k
          8.13   E_d = max( (1−a_Q)[a_G γ_G g_k + (1−a_G) γ_GP p_k] + a_Q ψ₀ γ_Q q_k ,
                            ξ (1−a_Q)[a_G γ_G g_k + (1−a_G) γ_GP p_k] + a_Q γ_Q q_k )
          8.14   E_d = max( (1−a_Q)[a_G γ_G g_k + (1−a_G) γ_GP p_k] ,
                            ξ (1−a_Q)[a_G γ_G g_k + (1−a_G) γ_GP p_k] + a_Q γ_Q q_k )

        and the design resistance z θ_R,k r_k / γ_M must equal E_d, so
        z = E_d γ_M / (θ_R,k r_k).
        """
        m, l = self.mode, self.load
        gk, pk, qk = m.G.char, self.model.P.char, l.Q.char
        gamma_Q = pf.gamma_Q_of(l.name)
        permanent = (1 - self.a_Q) * (self.a_G * pf.gamma_G * gk + (1 - self.a_G) * pf.gamma_GP * pk)
        variable = self.a_Q * gamma_Q * qk
        if equation == '8.12':
            E_d = permanent + variable
        elif equation == '8.13':
            E_d = max(permanent + l.psi_0 * variable, pf.xi * permanent + variable)
        elif equation == '8.14':
            E_d = max(permanent, pf.xi * permanent + variable)
        else:
            raise ValueError(f"Unknown design equation {equation!r}; use one of {EQUATIONS}")
        z = E_d * m.gamma_M / (m.theta_R.char * m.R.char)
        if not np.isfinite(z) or z <= 0:
            raise ValueError(f"design parameter z = {z!r} for {m.name} / {l.name}: "
                             "check the characteristic values and partial factors")
        return z

    # -- limit state ---------------------------------------------------------------
    def limit_state(self, z):
        """g(x) = z Θ_R R − Θ_E[(1−a_Q)(a_G G + (1−a_G) P) + a_Q Θ_Q Q] with its gradient."""
        aG, aQ = self.a_G, self.a_Q

        def g(x):
            tR, R, tE, G, P, tQ, Q = x
            return z * tR * R - tE * ((1 - aQ) * (aG * G + (1 - aG) * P) + aQ * tQ * Q)

        def grad(x):
            tR, R, tE, G, P, tQ, Q = x
            load = (1 - aQ) * (aG * G + (1 - aG) * P) + aQ * tQ * Q
            return np.array([z * R, z * tR, -load,
                             -tE * (1 - aQ) * aG, -tE * (1 - aQ) * (1 - aG),
                             -tE * aQ * Q, -tE * aQ * tQ])

        return LimitState(g, self.variables, grad,
                          name=f"{self.mode.name} / {self.load.name}, a_G={aG:.2f}, a_Q={aQ:.2f}")

    def reliability(self, pf: PartialFactors, equation='8.12', method='form'):
        """(β, FORMResult, z) for the given partial factors."""
        z = self.design_parameter(pf, equation)
        ls = self.limit_state(z)
        res = form(ls)
        if method == 'sorm':
            beta, _ = sorm(ls, res)
        elif method == 'form':
            beta = res.beta
        else:
            raise ValueError("method must be 'form' or 'sorm'")
        return beta, res, z


# ---- the sweep ---------------------------------------------------------------------

def evaluate(model: CalibrationModel, pf: PartialFactors, equation='8.12',
             method='form', situations=None, progress=None) -> pd.DataFrame:
    """β for every design situation. Returns a DataFrame with one row per
    situation (mode, load, a_G, a_Q, weight, z, beta, α_i) — the material
    for every table and plot of a calibration study."""
    situations = model.design_situations() if situations is None else situations
    rows = []
    for k, s in enumerate(situations):
        beta, res, z = s.reliability(pf, equation, method)
        row = {'mode': s.mode.name, 'load': s.load.name, 'a_G': s.a_G, 'a_Q': s.a_Q,
               'weight': s.weight, 'z': z, 'beta': beta, 'form_iterations': res.iterations,
               'converged': bool(res.converged)}
        row.update({f'alpha_{n}': a for n, a in zip(s.variable_names, res.alpha)})
        rows.append(row)
        if progress and k % 50 == 0:
            progress(k, len(situations))
    df = pd.DataFrame(rows)
    df['weight'] /= df['weight'].sum()          # Σ w = 1 (the AHG weights sum to 0.999)
    return df


def weighted_mean_beta(df: pd.DataFrame) -> float:
    return float((df['weight'] * df['beta']).sum() / df['weight'].sum())


def beta_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean β per failure mode (rows) and load (columns)."""
    return df.pivot_table(index='mode', columns='load', values='beta', aggfunc='mean', sort=False)
