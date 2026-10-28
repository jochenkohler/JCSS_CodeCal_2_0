"""
Learning mode: the six lessons as guided, interactive steps.

Each step has controls on the left, the picture and the numbers on the
right, a short explanation and a few things to try. The computations are
the same functions the lessons and the lab use.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st
from scipy.special import ndtr

from .. import ahg2021, plots, reliability
from ..calibrate import calibrate as run_calibration
from ..distributions import Gumbel, distribution
from ..model import DesignSituation, PartialFactors
from ..plots import Figure
from ..variables import rv, time_variant_load
from . import common as C

STEPS = [
    '1 · Random variables',
    '2 · FORM playground',
    '3 · One design situation',
    '4 · The sweep (status quo)',
    '5 · Calibration',
    '6 · Verification',
]


def _learning_spec():
    """The AHG model for the learning mode: 1 M Monte-Carlo draws keep the
    first build at a second or two."""
    if 'learning_spec' not in st.session_state:
        st.session_state['learning_spec'] = ahg2021.ahg_spec(n_mcs=1_000_000)
    return st.session_state['learning_spec']


def _background(text):
    with st.expander('Background', expanded=False):
        st.markdown(text)


def _try(items):
    with st.expander('Things to try'):
        for it in items:
            st.markdown(f'- {it}')


# ---- step 1 ----------------------------------------------------------------------------

def step_random_variables():
    st.subheader('Random variables and characteristic values')
    _background(r'''
Every quantity of a design — strength, model uncertainty, self-weight, climatic load — is a
random variable with a distribution, a mean and a coefficient of variation (CoV). The code
works with one *characteristic value* $x_k$ per quantity: a fractile (5 % for strengths, the
mean for permanent loads, the 98 % fractile of the **yearly** maximum for climatic loads).
All variables of the AHG model are normalised to mean 1, so $x_k$ is a ratio to the mean.

FORM (step 2) works in the space of standard-normal variables $u$; each variable is mapped by
$x = F^{-1}(\Phi(u))$. The right-hand picture shows that mapping.
''')
    left, right = st.columns([1, 2])
    with left:
        family = st.selectbox('Distribution', ['lognormal', 'normal', 'gumbel'], key='l1_family')
        mean = st.number_input('Mean', 0.01, 100.0, 1.0, 0.05, key='l1_mean')
        cov = st.number_input('CoV', 0.005, 1.5, 0.15, 0.01, key='l1_cov')
        frac = st.select_slider('Fractile for x_k', [0.01, 0.02, 0.05, 0.10, 0.50, 0.90, 0.95, 0.98, 0.99], 0.05,
                                key='l1_frac')
    with C.guarded('distribution') as g:
        d = distribution(family, mean, cov)
        xk = float(d.ppf(frac))
        with left:
            st.metric(f'x_k = {frac:.0%} fractile', f'{xk:.3f}', f'{xk / mean - 1:+.1%} vs mean', delta_color='off')
            st.caption('x_k / mean depends only on the CoV and the family.')
        with right:
            x = np.linspace(max(1e-6, mean * (1 - 4 * cov)), mean * (1 + 5 * cov), 500)
            fig = Figure(figsize=(8, 3.2))
            ax1, ax2 = fig.subplots(1, 2)
            ax1.plot(x, d.pdf(x), color=plots.CURRENT)
            ax1.axvline(xk, color=plots.TARGET, ls='--', label=f'x_k ({frac:.0%})')
            ax1.axvline(mean, color='grey', ls=':', label='mean')
            ax1.set_xlabel('x')
            ax1.set_ylabel('density')
            ax1.legend(fontsize=8)
            ax1.grid(alpha=0.3)
            u = np.linspace(-3.5, 3.5, 200)
            ax2.plot(u, d.x_of_u(u), color=plots.CURRENT)
            ax2.set_xlabel('u (standard normal)')
            ax2.set_ylabel('x = F⁻¹(Φ(u))')
            ax2.grid(alpha=0.3)
            fig.tight_layout()
            C.show(fig)
            tab = pd.DataFrame({'fractile': ['1 %', '5 %', '50 %', '95 %', '98 %', '99 %'],
                                'x': [d.ppf(p) for p in (0.01, 0.05, 0.5, 0.95, 0.98, 0.99)]})
            st.dataframe(tab.style.format({'x': '{:.3f}'}), hide_index=True, width='stretch')
    if g.failed:
        return

    st.markdown('#### Yearly maximum → T-year maximum, and the characteristic load')
    st.markdown('A Gumbel yearly maximum becomes a *T*-year maximum with the same σ and the mean shifted by '
                '(√6/π) σ ln T. The characteristic climatic load is a fractile of the yearly maximum of the '
                'load *times its model uncertainty* Θ_Q — simulated, as in the AHG model.')
    c1, c2, c3, c4 = st.columns(4)
    q_cov = c1.number_input('CoV of the 50-year maximum', 0.05, 0.6, 0.14, 0.01, key='l1_qcov')
    T = c2.select_slider('Reference period T (years)', [1, 5, 10, 25, 50, 100], 50, key='l1_T')
    tq_cov = c3.number_input('CoV of Θ_Q', 0.0, 0.6, 0.26, 0.01, key='l1_tqcov')
    tq_mean = c4.number_input('mean of Θ_Q', 0.5, 1.5, 0.97, 0.01, key='l1_tqmean')
    with C.guarded('load transformation') as g:
        q50 = Gumbel.from_moments(1.0, q_cov)
        q1 = q50.over_period(1 / 50)
        qT = q50.over_period(T / 50)
        theta = rv('Θ_Q', 'lognormal', tq_mean, max(tq_cov, 1e-6), char=1.0)
        rng = np.random.RandomState(1)
        Q, Qb = time_variant_load('Q', 1.0, q_cov, 50, 1, T, theta, 0.98, n_mcs=400_000, rng=rng)
        x = np.linspace(0, 2.2, 400)
        fig = Figure(figsize=(8, 3.0))
        ax = fig.subplots()
        ax.plot(x, q1.pdf(x), label='yearly maximum')
        ax.plot(x, qT.pdf(x), label=f'{T}-year maximum')
        ax.axvline(q1.ppf(0.98), color='k', ls='--', lw=1, label='98 % fractile of the yearly maximum (Q only)')
        ax.set_xlabel('load (relative to the mean 50-year maximum)')
        ax.set_ylabel('density')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        C.show(fig)
        c1, c2, c3 = st.columns(3)
        c1.metric(f'mean of the {T}-year maximum', f'{qT.mean:.3f}')
        c2.metric('q_k of Q alone (98 % yearly)', f'{q1.ppf(0.98):.3f}')
        qk_rel = Q.char * qT.mean          # Q.char is relative to the T-year mean; rescale to the 50-year mean
        c3.metric('q_k of Θ_Q·Q (simulated)', f'{qk_rel:.3f}',
                  help='98 % fractile of the yearly maximum of Θ_Q·Q, relative to the mean 50-year maximum.')
    _try([
        'Set the CoV of Θ_Q to 0: the simulated q_k equals the 98 % fractile of Q alone.',
        'Lognormal with CoV 0.20 against normal with CoV 0.20: which 5 % fractile is lower, and why?',
        'Period T = 1: the analysis would use the yearly maximum, with a larger CoV than the 50-year one.',
    ])


# ---- step 2 ----------------------------------------------------------------------------

def step_form():
    st.subheader('FORM on a two-variable limit state')
    _background(r'''
The reliability index is the distance from the origin to the failure surface $g(\mathbf u)=0$
in standard-normal space; the closest point is the *design point*, $P_f \approx \Phi(-\beta)$.
The HL-RF iteration finds it in a handful of steps. $\alpha = \mathbf u^*/\beta$ holds the
sensitivities: $\alpha_i^2$ is the share of variable $i$ in the uncertainty at the design point.

FORM is exact for a linear surface and optimistic where the surface bends towards the origin —
which happens for products of lognormal and Gumbel variables, the case of code calibration.
SORM (curvature correction) and importance sampling around the design point check it cheaply.
''')
    left, right = st.columns([1, 2])
    with left:
        kind = st.radio('Limit state', ['g = R − S', 'g = c − Θ·Q  (product)'], key='l2_kind')
        if kind.startswith('g = R'):
            famR = st.selectbox('R family', ['lognormal', 'normal'], key='l2_famR')
            meanR = st.number_input('R mean', 0.5, 10.0, 2.6, 0.1, key='l2_meanR')
            covR = st.number_input('R CoV', 0.01, 1.0, 0.15, 0.01, key='l2_covR')
            famS = st.selectbox('S family', ['gumbel', 'normal', 'lognormal'], key='l2_famS')
            covS = st.number_input('S CoV', 0.01, 1.0, 0.30, 0.01, key='l2_covS')
        else:
            c = st.number_input('fixed resistance c', 0.5, 10.0, 2.4, 0.1, key='l2_c')
            covT = st.number_input('Θ CoV (lognormal)', 0.01, 1.0, 0.26, 0.01, key='l2_covT')
            covQ = st.number_input('Q CoV (Gumbel)', 0.01, 1.0, 0.14, 0.01, key='l2_covQ')
        n_is = st.select_slider('importance-sampling samples', [2_000, 10_000, 50_000, 200_000], 10_000, key='l2_nis')
        do_mc = st.checkbox('also crude Monte Carlo (2 M samples, slower)', key='l2_mc')
    with C.guarded('FORM') as g:
        if kind.startswith('g = R'):
            R = rv('R', famR, meanR, covR, char=1.0)
            S = rv('S', famS, 1.0, covS, char=1.0)
            ls = reliability.LimitState(lambda x: x[0] - x[1], [R, S],
                                        lambda x: np.array([np.ones_like(x[0]), -np.ones_like(x[1])]), name='R − S')
        else:
            T_ = rv('Θ', 'lognormal', 1.0, covT, char=1.0)
            Q_ = rv('Q', 'gumbel', 1.0, covQ, char=1.0)
            ls = reliability.LimitState(lambda x: c - x[0] * x[1], [T_, Q_],
                                        lambda x: np.array([-x[1], -x[0]]), name='c − Θ·Q')
        res = reliability.form(ls)
        b_sorm, kappa = reliability.sorm(ls, res)
        isr = reliability.importance_sampling(ls, res, n=n_is, seed=1)
        with right:
            C.show(plots.plot_limit_state_2d(ls, res, lim=5, title=f'{ls.name} in u-space'))
        with left:
            st.metric('β (FORM)', f'{res.beta:.3f}', f'P_f ≈ {res.pf:.2e}', delta_color='off')
            if not res.converged:
                st.warning('FORM did not converge — the picture shows the last iterate.')
        rows = [('FORM', res.beta, None), ('SORM (Breitung)', b_sorm, None),
                (f'importance sampling, n = {n_is:,}', isr.beta, isr.se_beta)]
        if do_mc:
            t0 = time.time()
            mc = reliability.monte_carlo(ls, n=2_000_000, seed=1)
            rows.append((f'crude Monte Carlo, 2,000,000 samples ({time.time() - t0:.1f} s)', mc.beta, mc.se_beta))
        tab = pd.DataFrame([{'method': m, 'β': b, '± (std. error)': se} for m, b, se in rows])
        st.dataframe(tab.style.format({'β': '{:.4f}', '± (std. error)': '{:.4f}'}, na_rep=''), hide_index=True,
                     width='stretch')
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('**Sensitivities and design point**')
            st.dataframe(pd.DataFrame({'variable': [v.name for v in ls.variables],
                                       'α': res.alpha, 'α²': res.alpha ** 2, 'x*': res.x_star,
                                       'u*': res.u_star}).style.format('{:.3f}', subset=['α', 'α²', 'x*', 'u*']),
                         hide_index=True, width='stretch')
            st.caption(f'curvature κ at the design point: {kappa.round(4)}')
        with c2:
            C.show(plots.plot_form_history(res))
    _try([
        'R − S with a normal S: the surface is almost straight, FORM = sampling.',
        'The product c − Θ·Q: FORM is above SORM and sampling by a few hundredths — the calibration limit state behaves like this.',
        'Give R the larger CoV: α moves from the load to the resistance. EN 1990 fixes α_R = −0.8, α_E = 0.7.',
    ])


# ---- step 3 ----------------------------------------------------------------------------

def step_design_situation():
    st.subheader('The design equation and one design situation')
    _background(r'''
A design situation is a failure mode, a dominating variable load and the load proportions
$a_G$ (share of self-weight in the permanent load) and $a_Q$ (share of the variable load).
The design equation fixes how much resistance the engineer provides — the *design parameter*
$z$ — and FORM on

$g = z\,\Theta_R R - \Theta_E[(1-a_Q)(a_G G + (1-a_G) P) + a_Q \Theta_Q Q]$

gives the reliability that results. Change the partial factors and watch β follow.
''')
    spec = _learning_spec()
    with C.guarded('model') as g:
        model = C.build_model(spec)
    if g.failed:
        return
    left, right = st.columns([1, 2])
    with left:
        mode = st.selectbox('Failure mode', [m.name for m in model.failure_modes], key='l3_mode')
        load = st.selectbox('Variable load', [l.name for l in model.loads], key='l3_load')
        a_G = st.slider('a_G — self-weight share of the permanent load', 0.0, 1.0, 0.8, 0.05, key='l3_aG')
        a_Q = st.slider('a_Q — variable-load share', 0.0, 1.0, 0.5, 0.05, key='l3_aQ')
        equation = st.selectbox('Design equation', list(C.EQUATION_LABELS), format_func=C.EQUATION_LABELS.get,
                                key='l3_eq')
        method = st.radio('β method', ['form', 'sorm'], horizontal=True, key='l3_method',
                          format_func=str.upper)
    pf = C.factor_controls(spec, 'l3', with_xi=equation != '8.12')
    with C.guarded('design situation') as g:
        m = next(x for x in model.failure_modes if x.name == mode)
        l = next(x for x in model.loads if x.name == load)
        s = DesignSituation(m, l, a_G, a_Q, 1.0, model)
        beta, res, z = s.reliability(pf, equation, method)
        with right:
            c = st.columns(3)
            c[0].metric('design parameter z', f'{z:.3f}', help='mean resistance per unit mean load that the code demands')
            c[1].metric(f'β ({method.upper()})', f'{beta:.3f}', f'{beta - spec["beta_target"]:+.2f} vs β_t', delta_color='off')
            c[2].metric('P_f (50 a)', f'{ndtr(-beta):.2e}')
            if not res.converged:
                st.warning('FORM did not converge for this situation.')
            fig = Figure(figsize=(8, 3))
            ax1, ax2 = fig.subplots(1, 2)
            ax1.bar(s.variable_names, res.alpha ** 2, color=plots.CURRENT)
            ax1.set_ylabel('α²')
            ax1.set_title('who carries the uncertainty', fontsize=10)
            ax1.grid(axis='y', alpha=0.3)
            gq = np.linspace(1.0, 2.2, 13)
            for aq in sorted({0.3, 0.5, 0.8, round(a_Q, 2)}):
                sit = DesignSituation(m, l, a_G, aq, 1.0, model)
                betas = []
                for v in gq:
                    betas.append(sit.reliability(pf.with_values(['gamma_Q'], [v]), equation, method)[0])
                ax2.plot(gq, betas, '-', lw=2 if abs(aq - a_Q) < 1e-9 else 1, label=f'a_Q = {aq}')
            ax2.axhline(spec['beta_target'], color=plots.TARGET, ls='--')
            ax2.set_xlabel('γ_Q (all loads)')
            ax2.set_ylabel('β')
            ax2.set_title('β against γ_Q', fontsize=10)
            ax2.legend(fontsize=8)
            ax2.grid(alpha=0.3)
            fig.tight_layout()
            C.show(fig)
            st.dataframe(pd.DataFrame({'variable': s.variable_names,
                                       'distribution': [type(v.dist).__name__ for v in s.variables],
                                       'mean': [v.mean for v in s.variables], 'CoV': [v.cov for v in s.variables],
                                       'x_k': [v.char for v in s.variables], 'x* (design point)': res.x_star,
                                       'α': res.alpha}).style.format('{:.3f}', subset=['mean', 'CoV', 'x_k', 'x* (design point)', 'α']),
                         hide_index=True, width='stretch')
    _try([
        'Steel bending under wind at a_Q = 0.5: three variables carry almost everything — which ones?',
        'Raise a_Q to 0.8: β drops. The present factors favour permanent-load-dominated structures.',
        'Switch to eq. 8.13 or 8.14: z is smaller at every a_Q, and ξ starts to matter.',
    ])


# ---- step 4 ----------------------------------------------------------------------------

def step_sweep():
    st.subheader('The sweep: how reliable are structures designed with the present code?')
    _background(r'''
9 failure modes × 3 loads × 3 values of $a_G$ × 10 values of $a_Q$ = 810 design situations,
each with a weight $w = w_{mode} w_{load}/(n_{a_G} n_{a_Q})$. Evaluating all of them with
the present partial factors ($\gamma_G = 1.35$, $\gamma_Q = 1.5$) is the *status quo*
of the code. The weighted mean is the headline; the spread is the story.
''')
    spec = _learning_spec()
    c = st.columns(3)
    equation = c[0].selectbox('Design equation', list(C.EQUATION_LABELS), format_func=C.EQUATION_LABELS.get, key='l4_eq')
    method = c[1].radio('β method', ['form', 'sorm'], horizontal=True, key='l4_method', format_func=str.upper)
    grid = c[2].radio('Grid', ['full (3 × 10)', 'quick (1 × 5)'], horizontal=True, key='l4_grid')
    pf = C.factor_controls(spec, 'l4', with_xi=equation != '8.12')
    spec_run = dict(spec, n_aG=3 if grid.startswith('full') else 1, n_aQ=10 if grid.startswith('full') else 5)
    if st.button('Run the sweep', type='primary', key='l4_run'):
        with C.guarded('status-quo sweep') as g:
            with st.spinner(f'FORM for {C.situation_count(spec_run)} design situations…'):
                df = C.run_sweep(spec_run, pf, equation, method)
            st.session_state['l4_result'] = {'df': df, 'pf': pf, 'equation': equation, 'method': method, 'spec': spec_run}
    res = st.session_state.get('l4_result')
    if res:
        C.sweep_metrics(res['df'], spec['beta_target'])
        C.sweep_plots(res['df'], spec['beta_target'], 'l4')
        st.caption(f"eq. {res['equation']}, {res['method'].upper()}, "
                   + ', '.join(f'{C.gamma_label(k)} = {v:.3f}' for k, v in res['pf'].as_dict().items()))
    _try([
        'β falls with a_Q in every panel of "β against a_Q": variable-load-dominated structures are the least reliable.',
        'Snow-dominated designs sit lowest, imposed-load designs highest — the loads differ systematically.',
        'Sensitivities: the three model uncertainties together carry more than half of the reliability.',
    ])


# ---- step 5 ----------------------------------------------------------------------------

def step_calibration():
    st.subheader('Calibration: choosing the partial factors')
    _background(r'''
Minimise $M(\gamma) = \sum_k w_k (\beta_k(\gamma) - \beta_t)^2$ over the free partial factors.
One pair $(\gamma_G, \gamma_Q)$ cannot hit $\beta_t$ everywhere, so the optimum is a weighted
compromise; separate factors per load or a free ξ reduce the residual scatter at the price
of a more complicated code. Each evaluation of $M$ runs FORM for every situation.
''')
    spec = _learning_spec()
    c = st.columns(3)
    equation = c[0].selectbox('Design equation', list(C.EQUATION_LABELS), format_func=C.EQUATION_LABELS.get, key='l5_eq')
    method = c[1].radio('β method', ['form', 'sorm'], horizontal=True, key='l5_method', format_func=str.upper)
    grid = c[2].radio('Grid', ['full (3 × 10)', 'quick (1 × 5)'], horizontal=True, key='l5_grid',
                      help='The quick grid gives the result in seconds and is close to the full one.')
    options = ['gamma_G', 'gamma_GP', 'gamma_Q'] + [f'gamma_Q:{l["name"]}' for l in spec['loads']] + ['xi']
    default = ['gamma_G', 'gamma_Q'] + (['xi'] if equation != '8.12' else [])
    free = st.multiselect('Free parameters', options, default, format_func=C.FREE_PARAMETER_LABELS.get, key=f'l5_free_{equation}')
    beta_t = st.number_input('target β_t', 2.0, 6.0, float(spec['beta_target']), 0.1, key='l5_bt')
    problems = []
    if 'gamma_Q' in free and any(f.startswith('gamma_Q:') for f in free):
        problems.append('Choose either a common γ_Q or per-load factors, not both.')
    if 'xi' in free and equation == '8.12':
        problems.append('ξ does not appear in eq. 8.12 — it has no effect there.')
    if not free:
        problems.append('Choose at least one free parameter.')
    for p in problems:
        st.warning(p)
    spec_run = dict(spec, n_aG=3 if grid.startswith('full') else 1, n_aQ=10 if grid.startswith('full') else 5)
    n = C.situation_count(spec_run)
    st.caption(f'{n} design situations per evaluation; a calibration needs 15–60 evaluations '
               f'(≈ {C.estimate_seconds(n, method, 30):.0f} s).')
    if st.button('Calibrate', type='primary', key='l5_run', disabled=bool(problems)):
        with C.guarded('calibration') as g:
            model = C.build_model(spec_run)
            status = st.status('Calibrating…', expanded=True)
            bar = status.progress(0.0)
            text = status.empty()

            def progress(k, x, val):
                bar.progress(min(k / 60, 0.98))
                text.markdown(f'evaluation {k}: ' + ', '.join(f'{C.gamma_label(n)} = {v:.4f}' for n, v in zip(free, x))
                              + f' → M = {val:.4e}')
            res = run_calibration(model, tuple(free), equation=equation, method=method, beta_target=beta_t,
                                   start=C.default_factors(spec), verbose=False, progress=progress)
            bar.progress(1.0)
            status.update(label=f'Calibration finished after {len(res.history)} evaluations', state='complete', expanded=False)
            st.session_state['l5_result'] = {'res': res, 'equation': equation, 'method': method, 'spec': spec_run,
                                             'free': tuple(free)}
    out = st.session_state.get('l5_result')
    if out:
        res = out['res']
        if not res.optimizer.success:
            st.warning(f'The optimiser stopped without declaring convergence: {res.optimizer.message}. '
                       'The best point found is shown.')
        c = st.columns(len(res.free) + 2)
        for i, name in enumerate(res.free):
            c[i].metric(C.gamma_label(name), f'{res.factors.get(name):.3f}')
        c[-2].metric('weighted-mean β', f'{res.beta_after:.3f}', f'{res.beta_after - res.beta_before:+.3f} vs start')
        c[-1].metric('objective M', f'{res.objective:.4f}', f'{res.objective - res.history[0][1]:+.4f} vs start',
                     delta_color='inverse')
        C.factors_table(res.factors, res.free)
        C.sweep_plots(res.sweep_after, res.beta_target, 'l5', ref_df=res.sweep_before,
                      labels=('calibrated', 'start values'))
        with st.expander('Optimiser history'):
            C.show(plots.plot_calibration_history(res))
    _try([
        'γ_G and γ_Q for eq. 8.12: the optimum lowers γ_G and raises γ_Q — see the status quo of step 4.',
        'Per-load γ_Q: snow gets the largest factor, imposed load the smallest; the objective roughly halves.',
        'Target 4.3 (consequence class 3): how much do the factors rise?',
    ])


# ---- step 6 ----------------------------------------------------------------------------

def step_verification():
    st.subheader('How good is FORM? SORM and importance sampling over the model')
    _background(r'''
For a sample of the design situations, compare FORM with SORM and with importance sampling
(20 000 samples each, ± 0.004 in β). The calibration limit state contains products of
lognormal and Gumbel variables, so FORM is optimistic by a few hundredths — systematically.
''')
    spec = _learning_spec()
    c = st.columns(3)
    step = c[0].select_slider('every k-th situation', [1, 3, 9, 27, 81], 9, key='l6_step')
    equation = c[1].selectbox('Design equation', list(C.EQUATION_LABELS), format_func=C.EQUATION_LABELS.get, key='l6_eq')
    n_is = c[2].select_slider('importance-sampling samples', [5_000, 20_000, 100_000], 20_000, key='l6_nis')
    pf = C.factor_controls(spec, 'l6', with_xi=equation != '8.12')
    if st.button('Verify', type='primary', key='l6_run'):
        with C.guarded('verification') as g:
            model = C.build_model(spec)
            sits = model.design_situations()[::step]
            rows = []
            bar = st.progress(0.0, text=f'{len(sits)} situations')
            for i, s in enumerate(sits):
                z = s.design_parameter(pf, equation)
                ls = s.limit_state(z)
                r = reliability.form(ls)
                b_s, _ = reliability.sorm(ls, r)
                isr = reliability.importance_sampling(ls, r, n=n_is, seed=i)
                rows.append({'mode': s.mode.name, 'load': s.load.name, 'a_G': s.a_G, 'a_Q': s.a_Q,
                             'β FORM': r.beta, 'β SORM': b_s, 'β IS': isr.beta, 'se IS': isr.se_beta,
                             'dominant': s.variable_names[int(np.argmax(np.abs(r.alpha)))]})
                if i % 10 == 0:
                    bar.progress((i + 1) / len(sits))
            bar.empty()
            st.session_state['l6_result'] = pd.DataFrame(rows)
    ver = st.session_state.get('l6_result')
    if ver is not None:
        d_f, d_s = ver['β FORM'] - ver['β IS'], ver['β SORM'] - ver['β IS']
        c = st.columns(4)
        c[0].metric('FORM − IS, mean', f'{d_f.mean():+.4f}')
        c[1].metric('FORM − IS, max |·|', f'{d_f.abs().max():.4f}')
        c[2].metric('SORM − IS, mean', f'{d_s.mean():+.4f}')
        c[3].metric('within 2 s.e. of IS', f'FORM {np.mean(d_f.abs() < 2 * ver["se IS"]):.0%} · SORM {np.mean(d_s.abs() < 2 * ver["se IS"]):.0%}')
        fig = Figure(figsize=(8, 3.4))
        ax = fig.subplots()
        for load, grp in ver.groupby('load'):
            ax.plot(grp['β IS'], grp['β FORM'] - grp['β IS'], '.', ms=5, label=f'FORM − IS, {load}')
        ax.plot(ver['β IS'], d_s, 'k.', ms=3, alpha=0.5, label='SORM − IS')
        ax.axhline(0, color='grey', lw=0.8)
        ax.set_xlabel('β (importance sampling)')
        ax.set_ylabel('error in β')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        C.show(fig)
        st.dataframe(ver.groupby('dominant')[['β FORM', 'β SORM', 'β IS']].mean().assign(
            **{'FORM − IS': lambda t: t['β FORM'] - t['β IS'], 'SORM − IS': lambda t: t['β SORM'] - t['β IS']})
            .style.format('{:.3f}'), width='stretch')
        with st.expander('All verified situations'):
            st.dataframe(ver.style.format('{:.4f}', subset=['β FORM', 'β SORM', 'β IS', 'se IS']), hide_index=True,
                         width='stretch')
    _try([
        'Situations dominated by a lognormal variable (the model uncertainties, the strengths) show the largest FORM error.',
        'Rerun step 5 with the SORM method: the calibrated factors rise by 0.01–0.02.',
    ])


def render():
    step = st.sidebar.radio('Step', STEPS, key='learning_step')
    st.title('JCSS CodeCal 2.0 — learning mode')
    st.caption('Reliability-based calibration of partial factors, step by step. '
               'The lessons in `lessons/` tell the same story with more words and exercises.')
    {STEPS[0]: step_random_variables, STEPS[1]: step_form, STEPS[2]: step_design_situation,
     STEPS[3]: step_sweep, STEPS[4]: step_calibration, STEPS[5]: step_verification}[step]()
