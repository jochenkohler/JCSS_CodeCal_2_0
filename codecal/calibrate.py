"""
Calibration: choose the partial factors that bring the reliability of all
design situations as close as possible to the target.

Objective (reliability-based, the AHG choice):

    M(γ) = Σ_k w_k (β_k(γ) − β_t)²

over the design situations k with weights w_k (Σ w_k = 1). It is minimised
with SLSQP (sequential least squares) under simple bounds. Each objective
evaluation runs FORM for every situation, so a calibration with the full
AHG model (810 situations) takes a minute or two.

`free` names the parameters the optimiser may vary; everything else keeps
its start value:

    ('gamma_G', 'gamma_Q')                      one γ_G for all permanent load, one γ_Q   (AHG "dre = 1")
    ('gamma_G', 'gamma_GP', 'gamma_Q')          separate γ for self-weight and other P   ("dre = 2")
    ('gamma_G', 'gamma_Q:wind', 'gamma_Q:snow', 'gamma_Q:imposed')    ("dre = 3")
    add 'xi' for the two-combination equations 8.13 / 8.14.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize

from .model import CalibrationModel, PartialFactors, evaluate, weighted_mean_beta

DEFAULT_BOUNDS = {
    'gamma_G': (0.0, 1.5), 'gamma_GP': (0.0, 1.5),
    'gamma_Q': (1.0, 3.5), 'xi': (0.5, 1.2),
}


def bounds_for(names, equation='8.12'):
    out = []
    for n in names:
        key = 'gamma_Q' if n.startswith('gamma_Q') else n
        lo, hi = DEFAULT_BOUNDS[key]
        if key in ('gamma_G', 'gamma_GP') and equation != '8.12':
            lo = 1.0                      # with ξ free, γ_G below 1 is not meaningful
        out.append((lo, hi))
    return tuple(out)


def objective(df, beta_target):
    """Σ w (β − β_t)² for an evaluated sweep."""
    return float((df['weight'] * (df['beta'] - beta_target) ** 2).sum())


@dataclass
class CalibrationResult:
    factors: PartialFactors
    free: tuple
    objective: float
    beta_target: float
    beta_before: float
    beta_after: float
    sweep_before: object
    sweep_after: object
    history: list = field(default_factory=list)     # (x, objective) per evaluation
    optimizer: object = None

    def __str__(self):
        lines = [f"calibrated {', '.join(self.free)}  (β_t = {self.beta_target})"]
        for n in self.free:
            lines.append(f"  {n:18s} {self.factors.get(n):.3f}")
        lines.append(f"  weighted-mean β  {self.beta_before:.3f} → {self.beta_after:.3f}")
        lines.append(f"  objective        {self.objective:.3e}   ({len(self.history)} evaluations)")
        return '\n'.join(lines)


def calibrate(model: CalibrationModel, free=('gamma_G', 'gamma_Q'), start=None,
              equation='8.12', method='form', beta_target=None, bounds=None,
              ftol=1e-6, eps=1e-4, verbose=True) -> CalibrationResult:
    """Minimise Σ w (β − β_t)² over the parameters named in `free`."""
    start = PartialFactors() if start is None else start
    beta_target = model.beta_target if beta_target is None else beta_target
    bounds = bounds_for(free, equation) if bounds is None else bounds
    situations = model.design_situations()

    before = evaluate(model, start, equation, method, situations)
    history = []

    def fun(x):
        pf = start.with_values(free, x)
        df = evaluate(model, pf, equation, method, situations)
        val = objective(df, beta_target)
        history.append((np.array(x), val))
        if verbose:
            print(f"  eval {len(history):3d}: " +
                  ', '.join(f"{n}={v:.4f}" for n, v in zip(free, x)) + f"  M = {val:.3e}")
        return val

    x0 = [start.get(n) for n in free]
    res = minimize(fun, x0, method='SLSQP', bounds=bounds, options={'ftol': ftol, 'eps': eps})
    factors = start.with_values(free, res.x)
    after = evaluate(model, factors, equation, method, situations)
    return CalibrationResult(factors, tuple(free), float(res.fun), beta_target,
                             weighted_mean_beta(before), weighted_mean_beta(after),
                             before, after, history, res)
