"""Tests of the codecal package. Run with `pytest` from the project folder."""
import json
import os
import sys

import numpy as np
import pytest
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import codecal as cc                                    # noqa: E402
from codecal.reliability import curvatures               # noqa: E402


# ---- distributions -------------------------------------------------------------

@pytest.mark.parametrize('family,mean,cov', [('normal', 1.0, 0.1), ('lognormal', 1.15, 0.05),
                                             ('gumbel', 1.0, 0.14), ('lognormal', 10.0, 0.3)])
def test_moments_round_trip(family, mean, cov):
    d = cc.distribution(family, mean, cov)
    assert d.mean == pytest.approx(mean, rel=1e-12)
    assert d.cov == pytest.approx(cov, rel=1e-12)


def test_fractiles_against_scipy():
    ln = cc.Lognormal.from_moments(10.0, 0.3)
    assert ln.ppf(0.05) == pytest.approx(stats.lognorm.ppf(0.05, ln.zeta, scale=np.exp(ln.lam)))
    gu = cc.Gumbel.from_moments(1.0, 0.2)
    assert gu.ppf(0.98) == pytest.approx(stats.gumbel_r.ppf(0.98, loc=gu.b, scale=1 / gu.a))
    assert gu.cdf(gu.ppf(0.98)) == pytest.approx(0.98)
    no = cc.Normal.from_moments(1.0, 0.1)
    assert no.ppf(0.5) == 1.0


@pytest.mark.parametrize('family', ['normal', 'lognormal', 'gumbel'])
def test_transform_is_isoprobabilistic_and_derivative_matches(family):
    d = cc.distribution(family, 1.0, 0.2)
    u = np.array([-3.0, -1.0, 0.0, 1.5, 3.0, 6.0])
    x = d.x_of_u(u)
    np.testing.assert_allclose(d.cdf(x[:5]), stats.norm.cdf(u[:5]), rtol=1e-9)
    assert np.all(np.isfinite(x)) and np.all(np.diff(x) > 0)
    h = 1e-6
    fd = (d.x_of_u(u + h) - d.x_of_u(u - h)) / (2 * h)
    np.testing.assert_allclose(d.dx_du(u), fd, rtol=1e-6)


def test_gumbel_period_shift():
    yearly = cc.Gumbel.from_moments(0.6, 0.3)
    fifty = yearly.over_period(50)
    assert fifty.std == pytest.approx(yearly.std)
    assert fifty.mean == pytest.approx(yearly.mean + np.sqrt(6) / np.pi * yearly.std * np.log(50))
    # the 50-year maximum of X is the max of 50 yearly maxima: F_50 = F_1^50
    x = 1.3
    assert fifty.cdf(x) == pytest.approx(yearly.cdf(x) ** 50, rel=1e-10)


# ---- random variables ------------------------------------------------------------

def test_rv_fractile_and_pinned():
    a = cc.rv('a', 'lognormal', 1.0, 0.1, fractile=0.05)
    assert a.char == pytest.approx(a.dist.ppf(0.05))
    b = cc.rv('b', 'lognormal', 1.0, 0.05, char=0.83)
    assert b.char == 0.83 and b.note == 'pinned'
    with pytest.raises(ValueError):
        cc.rv('c', 'normal', 1.0, 0.1)


def test_time_variant_load_normalisation():
    theta = cc.rv('θ', 'lognormal', 1.0, 0.1, char=1.0)
    rng = np.random.RandomState(1)
    Q, Q_basic = cc.time_variant_load('Q', 1.0, 0.14, T_input=50, T_basic=1, T_ref=50,
                                      theta_Q=theta, fractile=0.98, n_mcs=200_000, rng=rng)
    assert Q.mean == pytest.approx(1.0) and Q.cov == pytest.approx(0.14)
    assert Q_basic.std == pytest.approx(0.14)
    assert Q_basic.mean == pytest.approx(1 + np.sqrt(6) / np.pi * 0.14 * np.log(1 / 50))
    # reference period 100 a: smaller CoV, same physical σ relative to the larger mean
    Q100, _ = cc.time_variant_load('Q', 1.0, 0.14, 50, 1, 100, theta, 0.98, n_mcs=200_000, rng=rng)
    mean100 = 1 + np.sqrt(6) / np.pi * 0.14 * np.log(2)
    assert Q100.cov == pytest.approx(0.14 / mean100)
    # T_ref shorter than the basic period: analysis over T_basic
    Q5, _ = cc.time_variant_load('Q', 1.0, 0.26, 50, 5, 1, theta, 0.99, char=1.35)
    assert Q5.mean == pytest.approx(1.0) and Q5.cov == pytest.approx(0.26 / (1 + np.sqrt(6) / np.pi * 0.26 * np.log(5 / 50)))


# ---- reliability methods ---------------------------------------------------------

def _rs_lognormal(beta_exact=3.8, covR=0.15, covS=0.3):
    """R, S lognormal: ln R − ln S is normal, so β = (λ_R − λ_S)/√(ζ_R² + ζ_S²)
    exactly. Choose λ_R such that β equals `beta_exact`."""
    zR, zS = np.sqrt(np.log(1 + covR ** 2)), np.sqrt(np.log(1 + covS ** 2))
    lamS = -0.5 * zS ** 2                               # mean of S = 1
    lamR = lamS + beta_exact * np.sqrt(zR ** 2 + zS ** 2)
    R = cc.RandomVariable('R', cc.Lognormal(lamR, zR), 1.0)
    S = cc.RandomVariable('S', cc.Lognormal(lamS, zS), 1.0)
    return cc.LimitState(lambda x: x[0] - x[1], [R, S],
                         lambda x: np.array([1.0, -1.0]) * np.ones_like(x[0]))


def test_form_exact_for_lognormal_r_minus_s():
    ls = _rs_lognormal(3.8)
    res = cc.form(ls)
    assert res.converged
    assert res.beta == pytest.approx(3.8, abs=1e-6)
    assert np.sum(res.alpha ** 2) == pytest.approx(1.0)
    assert res.alpha[0] < 0 < res.alpha[1]                 # resistance / load signs
    # numerical gradient gives the same design point
    ls_num = cc.LimitState(ls.g_x, ls.variables)
    assert cc.form(ls_num).beta == pytest.approx(3.8, abs=1e-5)


def test_sorm_and_sampling_agree_on_a_curved_limit_state():
    R = cc.rv('R', 'lognormal', 2.5, 0.2, char=1.0)
    S = cc.rv('S', 'gumbel', 1.0, 0.3, char=1.0)
    ls = cc.LimitState(lambda x: x[0] - x[1], [R, S])
    res = cc.form(ls)
    b_sorm, kappa = cc.sorm(ls, res)
    mc = cc.monte_carlo(ls, n=2_000_000, seed=0)
    isr = cc.importance_sampling(ls, res, n=100_000, seed=0)
    assert abs(isr.beta - mc.beta) < 3 * np.hypot(isr.se_beta, mc.se_beta) + 0.005
    assert abs(b_sorm - mc.beta) < abs(res.beta - mc.beta) + 0.01     # SORM not worse than FORM
    assert isr.se_beta < mc.se_beta                                     # and far cheaper


def test_sorm_is_exact_for_a_linear_limit_state():
    R = cc.rv('R', 'normal', 2.0, 0.1, char=1.0)
    S = cc.rv('S', 'normal', 1.0, 0.3, char=1.0)
    ls = cc.LimitState(lambda x: x[0] - x[1], [R, S])
    res = cc.form(ls)
    exact = (2.0 - 1.0) / np.hypot(0.2, 0.3)
    assert res.beta == pytest.approx(exact, abs=1e-8)
    assert np.all(np.abs(curvatures(ls, res)) < 1e-6)
    assert cc.sorm(ls, res)[0] == pytest.approx(exact, abs=1e-6)


# ---- the calibration model -----------------------------------------------------------

@pytest.fixture(scope='module')
def model():
    return cc.ahg_model(n_mcs=200_000)


def test_design_equations(model):
    s = model.design_situations()[0]
    pf = cc.PartialFactors(gamma_G=1.35, gamma_GP=1.35, gamma_Q=1.5, xi=0.85)
    m, l = s.mode, s.load
    perm = (1 - s.a_Q) * (s.a_G * 1.35 * m.G.char + (1 - s.a_G) * 1.35 * model.P.char)
    var = s.a_Q * 1.5 * l.Q.char
    f = m.gamma_M / (m.theta_R.char * m.R.char)
    assert s.design_parameter(pf, '8.12') == pytest.approx((perm + var) * f)
    assert s.design_parameter(pf, '8.13') == pytest.approx(max(perm + l.psi_0 * var, 0.85 * perm + var) * f)
    assert s.design_parameter(pf, '8.14') == pytest.approx(max(perm, 0.85 * perm + var) * f)
    with pytest.raises(ValueError):
        s.design_parameter(pf, '6.10')


def test_limit_state_gradient(model):
    s = model.design_situations()[5]
    ls = s.limit_state(2.0)
    u = np.array([-0.3, -0.5, 0.8, 0.1, 0.1, 1.2, 1.0])
    ls_num = cc.LimitState(ls.g_x, ls.variables)
    np.testing.assert_allclose(ls.grad_u(u), ls_num.grad_u(u), rtol=1e-6)


def test_partial_factors_free_parameters():
    pf = cc.PartialFactors()
    pf2 = pf.with_values(['gamma_G', 'gamma_Q'], [1.2, 1.6])
    assert pf2.gamma_G == 1.2 and pf2.gamma_Q == {'wind': 1.6, 'snow': 1.6, 'imposed': 1.6}
    assert pf2.gamma_GP == 1.2                      # common permanent factor (AHG "dre = 1")
    pf4 = pf.with_values(['gamma_G', 'gamma_GP'], [1.2, 1.3])
    assert (pf4.gamma_G, pf4.gamma_GP) == (1.2, 1.3)
    with pytest.raises(ValueError):
        pf.with_values(['gamma_X'], [1.0])
    pf3 = pf.with_values(['gamma_Q:wind', 'xi'], [1.7, 0.9])
    assert pf3.gamma_Q['wind'] == 1.7 and pf3.gamma_Q['snow'] == 1.5 and pf3.xi == 0.9
    assert pf3.get('gamma_Q:wind') == 1.7 and pf2.get('gamma_Q') == 1.6


def test_sweep_weights_and_table(model):
    small = cc.ahg_model(n_mcs=200_000, n_aQ=2, n_aG=1)
    df = cc.evaluate(small, cc.START_VALUES)
    assert len(df) == 9 * 3 * 2
    assert df['weight'].sum() == pytest.approx(1.0)
    tab = cc.beta_table(df)
    assert tab.shape == (9, 3)
    assert 2.5 < cc.weighted_mean_beta(df) < 4.5


def test_calibration_moves_towards_target():
    small = cc.ahg_model(n_mcs=200_000, n_aQ=3, n_aG=1)
    small.failure_modes = small.failure_modes[:3]
    res = cc.calibrate(small, ('gamma_G', 'gamma_Q'), equation='8.12', verbose=False, ftol=1e-5)
    assert res.objective <= res.history[0][1] + 1e-12
    assert abs(res.beta_after - 3.8) < abs(res.beta_before - 3.8) + 0.02
    res2 = cc.calibrate(small, ('xi', 'gamma_G', 'gamma_Q'), equation='8.13', verbose=False, ftol=1e-4)
    assert set(res2.factors.as_dict()) >= {'xi', 'gamma_G'}


def test_calibration_reproduces_workbench_optimum():
    """Eurocode-Calibration-Workbench, eq. 8.12, dre = 1, FORM, 1 M MCS draws:
    gamma_G = gamma_GP = 1.2156, gamma_Q = 1.7497, objective 0.136515."""
    model = cc.ahg_model(n_mcs=1_000_000)
    res = cc.calibrate(model, ('gamma_G', 'gamma_Q'), equation='8.12', verbose=False, ftol=1e-8, eps=1e-5)
    assert res.factors.gamma_G == pytest.approx(1.2156, abs=2e-3)
    assert res.factors.gamma_GP == res.factors.gamma_G
    assert res.factors.gamma_Q_of('wind') == pytest.approx(1.7497, abs=2e-3)
    assert res.objective == pytest.approx(0.136515, abs=2e-5)


# ---- reproduction of the verified workbench numbers -----------------------------------

def test_reproduces_workbench_results():
    """Status-quo β of the AHG model, eq. 8.12, γ_G = 1.35, γ_Q = 1.5, computed
    with the Eurocode-Calibration-Workbench (October 2026, seed 12345)."""
    with open(os.path.join(ROOT, 'tests', 'reference_workbench.json')) as f:
        ref = json.load(f)
    model = cc.ahg_model(n_mcs=ref['mcs_n'], seed=ref['seed'])
    for load, key in (('wind', 'Z1'), ('snow', 'Z2'), ('imposed', 'Z3')):
        Q = next(l for l in model.loads if l.name == load).Q
        assert Q.char == pytest.approx(ref['char'][key], abs=1e-9)
    df = cc.evaluate(model, cc.START_VALUES, '8.12')
    assert cc.weighted_mean_beta(df) == pytest.approx(ref['beta_weighted_mean'], abs=2e-6)
    tab = cc.beta_table(df)
    modes = ['X11', 'X21', 'X22', 'X41', 'X51', 'X61', 'X71', 'X81', 'X82']
    for i, m in enumerate(model.failure_modes):
        for load, key in (('wind', 'Z1'), ('snow', 'Z2'), ('imposed', 'Z3')):
            assert tab.loc[m.name, load] == pytest.approx(ref['table'][key][modes[i]], abs=2e-6)
    steel = df[df['mode'] == 'Steel bending']['beta'].to_numpy()
    np.testing.assert_allclose(steel, ref['betabin_X11'], atol=2e-6)
