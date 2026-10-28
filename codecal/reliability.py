"""
Reliability methods: FORM, SORM, crude Monte Carlo and importance sampling.

Everything here works on a `LimitState`: a function g(x) of independent
random variables, failure for g ≤ 0. The methods operate in the space of
standard-normal variables u, reached variable by variable through
x_i = F_i⁻¹(Φ(u_i)) (`RandomVariable.x_of_u`).

FORM (first-order reliability method)
    finds the *design point* u*, the point of the failure surface g(u) = 0
    closest to the origin. Its distance β = ‖u*‖ is the reliability index
    and P_f ≈ Φ(−β). The unit vector α = u*/β holds the sensitivities: α_i²
    is the share of variable i in the total uncertainty at the design
    point (Σ α_i² = 1), and the sign tells whether the variable acts as a
    resistance (α_i < 0) or a load (α_i > 0).

    The iteration is the Hasofer–Lind / Rackwitz–Fiessler (HL-RF) scheme:
    linearise g at u_k, step to the design point of the linearised surface,
    repeat:   α_k = −∇g(u_k)/‖∇g(u_k)‖
              u_{k+1} = [α_k·u_k + g(u_k)/‖∇g(u_k)‖] α_k

SORM (second-order, Breitung 1984)
    corrects P_f for the curvature of the failure surface at u*:
    P_f ≈ Φ(−β) Π_i (1 + β κ_i)^(−1/2), κ_i the main curvatures. For the
    product-type limit state of code calibration FORM is on the unsafe side
    by a few hundredths in β; SORM removes most of that.

Monte Carlo
    draws u ~ N(0, I) and counts failures. Needs ~100/P_f samples for a
    10 % standard error — millions for β ≈ 4. Importance sampling centred
    at the FORM design point needs a few thousand instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import special


class LimitState:
    """g(x) with its variables; gradient analytic (`grad`) or numerical."""

    def __init__(self, g, variables, grad=None, name='g'):
        self.g_x = g                    # g(x) with x = list/array of physical values
        self.grad_x = grad              # ∂g/∂x (optional)
        self.variables = list(variables)
        self.name = name

    @property
    def n(self):
        return len(self.variables)

    def x_of_u(self, u):
        return np.array([v.x_of_u(ui) for v, ui in zip(self.variables, u)])

    def g_u(self, u):
        return self.g_x(self.x_of_u(u))

    def grad_u(self, u, h=1e-6):
        """∂g/∂u by the chain rule ∂g/∂u_i = ∂g/∂x_i · dx_i/du_i."""
        x = self.x_of_u(u)
        dxdu = np.array([v.dx_du(ui) for v, ui in zip(self.variables, u)])
        if self.grad_x is not None:
            return np.asarray(self.grad_x(x), dtype=float) * dxdu
        # central differences in x
        gx = np.empty(self.n)
        for i in range(self.n):
            e = np.zeros(self.n)
            e[i] = h * max(1.0, abs(x[i]))
            gx[i] = (self.g_x(x + e) - self.g_x(x - e)) / (2 * e[i])
        return gx * dxdu


@dataclass
class FORMResult:
    beta: float
    alpha: np.ndarray
    u_star: np.ndarray
    x_star: np.ndarray
    iterations: int
    converged: bool
    history: list = field(default_factory=list)     # (β_k, g(u_k)) per iteration

    @property
    def pf(self):
        return float(special.ndtr(-self.beta))

    def sensitivities(self, names=None):
        """α_i² per variable, as a dict."""
        names = names or [f'x{i}' for i in range(len(self.alpha))]
        return {n: round(float(a) ** 2, 4) for n, a in zip(names, self.alpha)}


def form(ls: LimitState, u0=None, tol=1e-6, maxiter=100) -> FORMResult:
    """FORM by the HL-RF iteration. Converged when the step ‖u_{k+1} − u_k‖
    is below `tol` and |g| is small relative to ‖∇g‖."""
    n = ls.n
    u = np.zeros(n) if u0 is None else np.asarray(u0, dtype=float)
    history = []
    for k in range(1, maxiter + 1):
        g = ls.g_u(u)
        grad = ls.grad_u(u)
        norm = np.linalg.norm(grad)
        if norm == 0:
            raise RuntimeError("FORM: zero gradient — g does not depend on u here.")
        alpha = -grad / norm
        beta = float(alpha @ u)                     # signed distance of u along α
        history.append((beta, float(g)))
        u_new = (alpha @ u + g / norm) * alpha     # HL-RF step
        step = np.linalg.norm(u_new - u)
        u = u_new
        if step < tol and abs(g) / norm < tol:       # |g|/‖∇g‖ = distance to the surface
            g = ls.g_u(u)
            grad = ls.grad_u(u)
            alpha = -grad / np.linalg.norm(grad)
            beta = float(alpha @ u)
            return FORMResult(beta, alpha, u, ls.x_of_u(u), k, True, history)
    return FORMResult(beta, alpha, u, ls.x_of_u(u), maxiter, False, history)


def hessian_u(ls: LimitState, u, h=1e-4):
    """Hessian of g in u-space by central differences of the gradient."""
    n = len(u)
    H = np.zeros((n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = h
        H[:, j] = (ls.grad_u(u + e) - ls.grad_u(u - e)) / (2 * h)
    return 0.5 * (H + H.T)


def curvatures(ls: LimitState, res: FORMResult):
    """Main curvatures κ_i of the failure surface at the design point, in
    the convention g ≈ β + ½ Σ κ_i v_i² − v_n in coordinates rotated so
    that v_n points along α (κ_i > 0: surface bends away from the origin,
    failure domain smaller than FORM's half-space)."""
    alpha = res.alpha
    n = len(alpha)
    grad_norm = np.linalg.norm(ls.grad_u(res.u_star))
    H = hessian_u(ls, res.u_star)
    # orthonormal basis with alpha last (Gram–Schmidt)
    basis = [alpha]
    for i in range(n):
        v = np.zeros(n)
        v[i] = 1.0
        for b in basis:
            v = v - (v @ b) * b
        if np.linalg.norm(v) > 1e-10:
            basis.append(v / np.linalg.norm(v))
        if len(basis) == n:
            break
    R = np.array(basis[1:] + [basis[0]])
    A = R @ H @ R.T / grad_norm
    return np.linalg.eigvalsh(A[:-1, :-1])


def sorm(ls: LimitState, res: FORMResult):
    """Breitung's SORM: returns (β_SORM, κ). Falls back to β_FORM if
    1 + β κ_i ≤ 0 for some i (formula not applicable)."""
    kappa = curvatures(ls, res)
    factors = 1.0 + res.beta * kappa
    if np.any(factors <= 0):
        return res.beta, kappa
    log_pf = special.log_ndtr(-res.beta) - 0.5 * np.sum(np.log(factors))
    return float(-special.ndtri(np.exp(log_pf))), kappa


@dataclass
class SamplingResult:
    pf: float
    beta: float
    se_beta: float          # standard error of β (delta method)
    n: int
    failures: float         # number (crude) or effective weight sum (IS)


def monte_carlo(ls: LimitState, n=1_000_000, seed=0, batch=200_000) -> SamplingResult:
    """Crude Monte Carlo in u-space."""
    rng = np.random.default_rng(seed)
    n = int(n)
    nfail = 0
    done = 0
    while done < n:
        m = min(batch, n - done)
        U = rng.standard_normal((ls.n, m))
        X = np.array([v.x_of_u(U[i]) for i, v in enumerate(ls.variables)])
        nfail += int(np.count_nonzero(ls.g_x(X) <= 0))
        done += m
    pf = nfail / n
    if nfail == 0:
        return SamplingResult(0.0, np.inf, np.nan, n, 0)
    beta = float(-special.ndtri(pf))
    se = np.sqrt(pf * (1 - pf) / n) / _phi(beta)
    return SamplingResult(pf, beta, float(se), n, nfail)


def importance_sampling(ls: LimitState, res: FORMResult, n=20_000, seed=0) -> SamplingResult:
    """Importance sampling with a standard-normal density centred at the
    FORM design point: P_f = E_h[ I(g ≤ 0) φ(u)/h(u) ], with
    φ(u)/h(u) = exp(−u·u* + ‖u*‖²/2)."""
    rng = np.random.default_rng(seed)
    n = int(n)
    U = res.u_star[:, None] + rng.standard_normal((ls.n, n))
    X = np.array([v.x_of_u(U[i]) for i, v in enumerate(ls.variables)])
    fail = ls.g_x(X) <= 0
    w = np.exp(-(res.u_star @ U) + 0.5 * (res.u_star @ res.u_star))
    ind = w * fail
    pf = float(ind.mean())
    if pf == 0:
        return SamplingResult(0.0, np.inf, np.nan, n, 0.0)
    se_pf = ind.std(ddof=1) / np.sqrt(n)
    beta = float(-special.ndtri(pf))
    return SamplingResult(pf, beta, float(se_pf / _phi(beta)), n, float(ind.sum()))


def _phi(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2 * np.pi)
