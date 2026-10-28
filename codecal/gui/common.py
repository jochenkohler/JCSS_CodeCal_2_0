"""
Pieces shared by the learning and the lab mode of the Streamlit app:
cached model building and sweeps, a guard that turns exceptions into
readable messages, the result panels, snapshots.
"""

from __future__ import annotations

import contextlib
import copy
import json
import sys
import time
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from .. import model as modelmod, plots, spec as specmod
from ..calibrate import objective
from ..model import PartialFactors

IN_BROWSER = sys.platform == 'emscripten'      # running under Pyodide / stlite (the GitHub Pages version)

EQUATION_LABELS = {
    '8.12': '8.12 — single combination (ex 6.10)',
    '8.13': '8.13 — max of (ψ₀ on Q) and (ξ on G)   (ex 6.10 a/b)',
    '8.14': '8.14 — max of (G only) and (ξ on G + Q)',
}
FREE_PARAMETER_LABELS = {
    'gamma_G': 'γ_G  (all permanent load, unless γ_G,P is free too)',
    'gamma_GP': 'γ_G,P  (other permanent load, separately)',
    'gamma_Q': 'γ_Q  (one factor for all variable loads)',
    'gamma_Q:wind': 'γ_Q,wind', 'gamma_Q:snow': 'γ_Q,snow', 'gamma_Q:imposed': 'γ_Q,imposed',
    'xi': 'ξ  (reduction of permanent load, eq. 8.13 / 8.14)',
}
GAMMA_SHORT = {'gamma_G': 'γ_G', 'gamma_GP': 'γ_G,P', 'gamma_Q': 'γ_Q', 'xi': 'ξ'}


def gamma_label(name):
    if name.startswith('gamma_Q:'):
        return 'γ_Q,' + name.split(':', 1)[1]
    return GAMMA_SHORT.get(name, name)


# ---- figures ---------------------------------------------------------------------

def show(fig):
    """Display a Figure made by codecal.plots."""
    st.pyplot(fig, width='stretch')


# ---- error handling ----------------------------------------------------------------

@contextlib.contextmanager
def guarded(what: str):
    """Run a block; on failure show a readable error instead of a traceback.

        with guarded('status-quo sweep') as ok:
            ...
        if ok.failed: return
    """
    state = type('Guard', (), {'failed': False})()
    try:
        yield state
    except specmod.ModelSpecError as e:
        state.failed = True
        st.error(f"The model is not valid — {what} was not run.")
        for p in e.problems:
            st.markdown(f"- {p}")
    except (ValueError, RuntimeError, ZeroDivisionError, FloatingPointError, KeyError,
            np.linalg.LinAlgError) as e:
        state.failed = True
        st.error(f"{what} failed: {e}")
        with st.expander("Details"):
            st.code(traceback.format_exc())
    except Exception as e:      # noqa: BLE001 — last resort, never a blank page
        state.failed = True
        st.error(f"Unexpected error during {what}: {type(e).__name__}: {e}")
        with st.expander("Details (please report)"):
            st.code(traceback.format_exc())


# ---- cached computations -----------------------------------------------------------

@st.cache_resource(show_spinner=False, max_entries=16)
def _model_cached(signature: str):
    return specmod.build_model(json.loads(signature))


def build_model(spec):
    """Model for a spec, cached by its canonical JSON (the Monte-Carlo
    characteristic values take seconds at 10 M draws)."""
    return _model_cached(specmod.spec_signature(spec))


@st.cache_data(show_spinner=False, max_entries=64)
def _sweep_cached(signature: str, pf_json: str, equation: str, method: str) -> pd.DataFrame:
    model = _model_cached(signature)
    pf = pf_from_dict(json.loads(pf_json))
    return modelmod.evaluate(model, pf, equation, method)


def run_sweep(spec, pf: PartialFactors, equation='8.12', method='form') -> pd.DataFrame:
    """β for every design situation (cached)."""
    return _sweep_cached(specmod.spec_signature(spec), json.dumps(pf.as_dict(), sort_keys=True),
                         equation, method)


def pf_from_dict(d: dict) -> PartialFactors:
    gq = {k.split(':', 1)[1]: float(v) for k, v in d.items() if k.startswith('gamma_Q:')}
    if not gq:
        gq = float(d.get('gamma_Q', 1.5))
    return PartialFactors(gamma_G=float(d.get('gamma_G', 1.35)), gamma_GP=float(d.get('gamma_GP', d.get('gamma_G', 1.35))),
                          gamma_Q=gq, xi=float(d.get('xi', 0.85)))


def default_factors(spec) -> PartialFactors:
    """Eurocode start values for the loads of a spec."""
    loads = [l['name'] for l in spec['loads'] if l.get('include', True)]
    return PartialFactors(1.35, 1.35, {n: 1.5 for n in loads}, 0.85)


# ---- partial-factor controls ---------------------------------------------------------

def factor_controls(spec, key: str, per_load=False, with_xi=True) -> PartialFactors:
    """Number inputs for the partial factors; returns a PartialFactors."""
    loads = [l['name'] for l in spec['loads'] if l.get('include', True)]
    c = st.columns(4 if not per_load else 3 + len(loads))
    gG = c[0].number_input('γ_G', 0.0, 2.5, 1.35, 0.05, key=f'{key}_gG', format='%.3f',
                           help='Partial factor on self-weight (and on the other permanent load, unless differentiated).')
    gGP = c[1].number_input('γ_G,P', 0.0, 2.5, 1.35, 0.05, key=f'{key}_gGP', format='%.3f',
                            help='Partial factor on the other permanent load.')
    if per_load:
        gq = {n: c[2 + i].number_input(f'γ_Q,{n}', 0.5, 4.0, 1.5, 0.05, key=f'{key}_gQ_{n}', format='%.3f')
              for i, n in enumerate(loads)}
        xi_col = c[2 + len(loads)]
    else:
        gq_common = c[2].number_input('γ_Q', 0.5, 4.0, 1.5, 0.05, key=f'{key}_gQ', format='%.3f',
                                      help='Partial factor on the variable load (same for all loads).')
        gq = {n: gq_common for n in loads}
        xi_col = c[3]
    xi = xi_col.number_input('ξ', 0.3, 1.5, 0.85, 0.05, key=f'{key}_xi', format='%.3f',
                             help='Reduction of the permanent load in the second combination of eq. 8.13 / 8.14.',
                             disabled=not with_xi)
    return PartialFactors(gG, gGP, gq, xi)


# ---- result panels -------------------------------------------------------------------

def sweep_metrics(df: pd.DataFrame, beta_target: float, ref_df: pd.DataFrame | None = None):
    wm = modelmod.weighted_mean_beta(df)
    cols = st.columns(4)
    delta = None if ref_df is None else f"{wm - modelmod.weighted_mean_beta(ref_df):+.3f} vs reference"
    cols[0].metric('weighted-mean β', f'{wm:.3f}', delta)
    cols[1].metric('target β_t', f'{beta_target:.2f}', f'{wm - beta_target:+.3f}', delta_color='off')
    cols[2].metric('min / max β', f'{df.beta.min():.2f} / {df.beta.max():.2f}')
    cols[3].metric('objective Σ w (β − β_t)²', f'{objective(df, beta_target):.4f}')
    bad = int((~df['converged']).sum()) if 'converged' in df else 0
    if bad:
        st.warning(f"FORM did not converge for {bad} of {len(df)} design situations; "
                   "their β values are the last iterate. Check the model for extreme CoVs.")
    if not np.isfinite(df['beta']).all():
        st.error("Some β values are not finite — the model is numerically unusable as it stands.")


def sweep_plots(df, beta_target, key, ref_df=None, labels=('current', 'reference')):
    view = st.radio('View', ['β per failure mode', 'β per load', 'β against a_Q', 'sensitivities α²',
                             'table: mean β per mode × load'],
                    horizontal=True, key=f'{key}_view', label_visibility='collapsed')
    if view == 'β per failure mode':
        show(plots.plot_beta_distribution(df, 'mode', beta_target, ref_df, labels))
    elif view == 'β per load':
        show(plots.plot_beta_distribution(df, 'load', beta_target, ref_df, labels))
    elif view == 'β against a_Q':
        show(plots.plot_beta_vs_aQ(df, beta_target, ref_df))
    elif view == 'sensitivities α²':
        show(plots.plot_sensitivities(df))
    else:
        tab = modelmod.beta_table(df)
        st.dataframe(tab.style.format('{:.2f}').background_gradient(cmap='RdYlGn', axis=None,
                                                                    vmin=beta_target - 1.3, vmax=beta_target + 0.7),
                     width='stretch')
        if ref_df is not None:
            st.caption('Reference (mean β per mode × load):')
            st.dataframe(modelmod.beta_table(ref_df).style.format('{:.2f}'), width='stretch')


def factors_table(pf: PartialFactors, free=None):
    d = pf.as_dict()
    rows = [{'factor': gamma_label(k), 'value': round(v, 4),
             'calibrated': '✓' if free and (k in free or (k.startswith('gamma_Q:') and 'gamma_Q' in free)
                                            or (k == 'gamma_GP' and 'gamma_G' in free and 'gamma_GP' not in free)) else ''}
            for k, v in d.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')


# ---- snapshots -------------------------------------------------------------------------

def snapshots() -> list:
    return st.session_state.setdefault('snapshots', [])


def add_snapshot(name, kind, df, pf, equation, method, spec, extra=None):
    snap = {'name': name.strip() or f'{kind} {len(snapshots()) + 1}', 'kind': kind,
            'df': df.copy(), 'factors': pf.as_dict(), 'equation': equation, 'method': method,
            'beta_mean': modelmod.weighted_mean_beta(df), 'beta_target': spec['beta_target'],
            'model': spec.get('name', ''), 'spec': copy.deepcopy(spec), 'time': time.strftime('%H:%M:%S')}
    snap.update(extra or {})
    snapshots().append(snap)
    return snap


def snapshot_panel(df, pf, equation, method, spec, kind, key, extra=None):
    """'Save as snapshot' row under a result."""
    c1, c2 = st.columns([3, 1])
    name = c1.text_input('Snapshot name', key=f'{key}_snapname', placeholder=f'e.g. "{kind}, {equation}"',
                         label_visibility='collapsed')
    if c2.button('Save snapshot', key=f'{key}_snapbtn', width='stretch'):
        snap = add_snapshot(name, kind, df, pf, equation, method, spec, extra)
        st.success(f"Saved snapshot “{snap['name']}” — compare it in the Compare tab.")


def snapshot_table():
    rows = []
    for s in snapshots():
        rows.append({'name': s['name'], 'kind': s['kind'], 'equation': s['equation'], 'method': s['method'],
                     'weighted-mean β': round(s['beta_mean'], 3),
                     'factors': ', '.join(f'{gamma_label(k)}={v:.3f}' for k, v in s['factors'].items()),
                     'model': s['model'], 'saved': s['time']})
    return pd.DataFrame(rows)


# ---- small helpers ---------------------------------------------------------------------

def situation_count(spec) -> int:
    nm = sum(1 for m in spec['modes'] if m.get('include', True))
    nl = sum(1 for l in spec['loads'] if l.get('include', True))
    return nm * nl * int(spec['n_aQ']) * int(spec['n_aG'])


def estimate_seconds(n_situations, method='form', evaluations=1):
    per = 0.0015 if method == 'form' else 0.0026       # measured on a laptop
    return per * n_situations * evaluations


def model_overview(model):
    st.caption(model.description)
    c = st.columns(4)
    c[0].metric('failure modes', len(model.failure_modes))
    c[1].metric('loads', len(model.loads))
    c[2].metric('design situations', len(model.failure_modes) * len(model.loads) * model.n_aQ * model.n_aG)
    c[3].metric('reference period', f'{model.T_ref:g} a')
    with st.expander('Random variables of the model'):
        rows = []
        for m in model.failure_modes:
            for v in (m.theta_R, m.R, m.G):
                rows.append({'group': m.name, 'variable': v.name, 'family': type(v.dist).__name__,
                             'mean': round(v.mean, 3), 'CoV': round(v.cov, 3), 'x_k': round(v.char, 3), 'note': v.note})
        for l in model.loads:
            for v in (l.theta_Q, l.Q):
                rows.append({'group': l.name, 'variable': v.name, 'family': type(v.dist).__name__,
                             'mean': round(v.mean, 3), 'CoV': round(v.cov, 3), 'x_k': round(v.char, 3), 'note': v.note})
        for v in (model.theta_E, model.P):
            rows.append({'group': 'all', 'variable': v.name, 'family': type(v.dist).__name__,
                         'mean': round(v.mean, 3), 'CoV': round(v.cov, 3), 'x_k': round(v.char, 3), 'note': v.note})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')
