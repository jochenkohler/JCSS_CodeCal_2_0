"""
Lab mode: the research tool.

Tabs: Model (edit every number, save/load JSON) · Status quo · Calibration ·
Compare (snapshots) · Sensitivity (vary one model parameter) · Verification.
The model lives in st.session_state as a plain dictionary (codecal.spec):
'lab_base' is what the editors were opened with (replaced on load / reset,
which bumps 'lab_spec_version' and re-keys the widgets), 'lab_spec' is the
live model — base plus the edits in the tables and the global inputs. Every
run validates it first and reports problems in words.
"""

from __future__ import annotations

import copy
import math
import re

import numpy as np
import pandas as pd
import streamlit as st

from .. import ahg2021, model as modelmod, plots, reliability, spec as specmod
from ..calibrate import bounds_for, calibrate as run_calibration, objective
from ..plots import Figure
from . import common as C

MODE_COLUMNS = ['include', 'name', 'Θ_R mean', 'Θ_R CoV', 'R CoV', 'R x_k', 'R fractile', 'γ_M',
                'G CoV', 'G x_k', 'G fractile', 'a_Q min', 'a_Q max', 'a_G min', 'a_G max', 'weight']
LOAD_COLUMNS = ['include', 'name', 'Θ_Q mean', 'Θ_Q CoV', 'Q CoV (T_input)', 'T_input', 'T_basic', 'fractile',
                'q_k pinned', 'ψ₀', 'weight']


# ---- spec ↔ tables ----------------------------------------------------------------------

def _base():
    """The spec the editors were opened with."""
    if 'lab_base' not in st.session_state:
        _set_spec(ahg2021.ahg_spec(n_mcs=1_000_000))
    return st.session_state['lab_base']


def _spec():
    """The live spec (base + the edits made in the Model tab)."""
    _base()
    return st.session_state['lab_spec']


def _set_spec(spec):
    """Replace the model (load, reset): new base, new widget keys."""
    st.session_state['lab_base'] = copy.deepcopy(spec)
    st.session_state['lab_spec'] = copy.deepcopy(spec)
    st.session_state['lab_spec_version'] = st.session_state.get('lab_spec_version', -1) + 1


def _numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Every column but 'include' and 'name' as float (None → NaN), so the
    editor shows number cells and empty cells for missing values."""
    for c in df.columns:
        if c not in ('include', 'name'):
            df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    return df


def modes_table(spec) -> pd.DataFrame:
    rows = []
    for m in spec['modes']:
        rows.append({'include': bool(m.get('include', True)), 'name': m['name'],
                     'Θ_R mean': m['theta_R'].get('mean'), 'Θ_R CoV': m['theta_R'].get('cov'),
                     'R CoV': m['R'].get('cov'), 'R x_k': m['R'].get('char'), 'R fractile': m['R'].get('fractile'),
                     'γ_M': m.get('gamma_M'), 'G CoV': m['G'].get('cov'), 'G x_k': m['G'].get('char'),
                     'G fractile': m['G'].get('fractile'),
                     'a_Q min': m['a_Q'][0], 'a_Q max': m['a_Q'][1], 'a_G min': m['a_G'][0], 'a_G max': m['a_G'][1],
                     'weight': m.get('weight')})
    return _numeric(pd.DataFrame(rows, columns=MODE_COLUMNS))


def loads_table(spec) -> pd.DataFrame:
    rows = []
    for l in spec['loads']:
        rows.append({'include': bool(l.get('include', True)), 'name': l['name'],
                     'Θ_Q mean': l['theta_Q'].get('mean'), 'Θ_Q CoV': l['theta_Q'].get('cov'),
                     'Q CoV (T_input)': l.get('cov'), 'T_input': l.get('T_input'), 'T_basic': l.get('T_basic'),
                     'fractile': l.get('fractile'), 'q_k pinned': l.get('char'), 'ψ₀': l.get('psi_0'),
                     'weight': l.get('weight')})
    return _numeric(pd.DataFrame(rows, columns=LOAD_COLUMNS))


def _opt(v):
    """Cell → float or None (empty cells arrive as NaN/None)."""
    if v is None or (isinstance(v, float) and math.isnan(v)) or v == '':
        return None
    return float(v)


def _rv_from_cells(old, cov, char, fractile):
    d = dict(old)
    d['cov'] = _opt(cov)
    char, fractile = _opt(char), _opt(fractile)
    d.pop('char', None)
    d.pop('fractile', None)
    if char is not None and fractile is None:
        d['char'] = char
    elif fractile is not None and char is None:
        d['fractile'] = fractile
    else:                                  # both or neither: keep both so validation reports it
        if char is not None:
            d['char'] = char
        if fractile is not None:
            d['fractile'] = fractile
    return d


def _base_entry(old: list, index, name: str, fresh: dict) -> dict:
    """The old entry an edited row belongs to: by its row index (the editor
    keeps the index of unchanged rows), else by name, else a fresh one."""
    if isinstance(index, (int, np.integer)) and 0 <= index < len(old):
        return copy.deepcopy(old[index])
    for e in old:
        if e['name'] == name:
            return copy.deepcopy(e)
    return copy.deepcopy(fresh)


def _fresh_mode(name):
    return {'name': name, 'theta_R': {'name': f'Θ_R {name}', 'family': 'lognormal', 'mean': 1.0, 'cov': 0.1,
                                      'char': 1.0, 'note': 'by definition'},
            'R': {'name': f'Resistance {name}', 'family': 'lognormal', 'mean': 1.0, 'cov': 0.1, 'fractile': 0.05},
            'gamma_M': 1.0,
            'G': {'name': f'Self-weight {name}', 'family': 'normal', 'mean': 1.0, 'cov': 0.05, 'fractile': 0.5},
            'a_Q': [0.1, 0.8], 'a_G': [0.6, 1.0], 'weight': 0.0, 'include': True}


def _fresh_load(name):
    return {'name': name, 'theta_Q': {'name': f'{name} model uncertainty', 'family': 'lognormal', 'mean': 1.0,
                                      'cov': 0.1, 'char': 1.0, 'note': 'by definition'},
            'mean': 1.0, 'cov': 0.2, 'T_input': 50, 'T_basic': 1, 'fractile': 0.98, 'char': None,
            'psi_0': 0.6, 'weight': 0.0, 'include': True}


def tables_to_spec(spec, modes_df, loads_df) -> dict:
    """Rebuild the spec from the edited tables (other fields untouched)."""
    new = copy.deepcopy(spec)
    modes = []
    for idx, r in modes_df.iterrows():
        name = str(r['name']).strip() if pd.notna(r['name']) else ''
        base = _base_entry(spec['modes'], idx, name, _fresh_mode(name or 'new mode'))
        base['name'] = name
        base['include'] = bool(r['include']) if pd.notna(r['include']) else True
        base['theta_R'] = dict(base['theta_R'], mean=_opt(r['Θ_R mean']), cov=_opt(r['Θ_R CoV']))
        base['R'] = _rv_from_cells(base['R'], r['R CoV'], r['R x_k'], r['R fractile'])
        base['G'] = _rv_from_cells(base['G'], r['G CoV'], r['G x_k'], r['G fractile'])
        base['gamma_M'] = _opt(r['γ_M'])
        base['a_Q'] = [_opt(r['a_Q min']), _opt(r['a_Q max'])]
        base['a_G'] = [_opt(r['a_G min']), _opt(r['a_G max'])]
        base['weight'] = _opt(r['weight'])
        modes.append(base)
    loads = []
    for idx, r in loads_df.iterrows():
        name = str(r['name']).strip() if pd.notna(r['name']) else ''
        base = _base_entry(spec['loads'], idx, name, _fresh_load(name or 'new load'))
        base['name'] = name
        base['include'] = bool(r['include']) if pd.notna(r['include']) else True
        base['theta_Q'] = dict(base['theta_Q'], mean=_opt(r['Θ_Q mean']), cov=_opt(r['Θ_Q CoV']))
        base['cov'] = _opt(r['Q CoV (T_input)'])
        base['T_input'] = _opt(r['T_input'])
        base['T_basic'] = _opt(r['T_basic'])
        base['fractile'] = _opt(r['fractile'])
        base['char'] = _opt(r['q_k pinned'])
        base['psi_0'] = _opt(r['ψ₀'])
        base['weight'] = _opt(r['weight'])
        loads.append(base)
    new['modes'], new['loads'] = modes, loads
    return new


# ---- Model tab ---------------------------------------------------------------------------

def _num_input(col, label, lo, hi, value, step, key, **kw):
    """number_input whose initial value is clamped into [lo, hi] — a loaded
    model may hold values outside the range the widget offers."""
    v = float(value) if value is not None and np.isfinite(float(value)) else lo
    if not lo <= v <= hi:
        st.warning(f'{label}: the model value {value!r} lies outside the range offered here; clamped.')
    return col.number_input(label, lo, hi, min(max(v, lo), hi), step, key=key, **kw)


def tab_model():
    spec = _base()
    v = st.session_state['lab_spec_version']
    if flash := st.session_state.pop('lab_flash', None):
        st.success(flash)
    st.markdown('Edit any cell; the tables are read back into the model as you type. '
                'Leave **x_k** empty to use the fractile, or the fractile empty to pin x_k. '
                'Untick **include** to leave a failure mode or load out.')

    c = st.columns([2, 1, 1, 1])
    with c[0]:
        name = st.text_input('Model name', spec.get('name', ''), key=f'lab_name_{v}')
    with c[1]:
        up = st.file_uploader('Load model (JSON)', type=['json'], key=f'lab_upload_{v}', label_visibility='visible')
    save_slot = c[2].empty()                 # filled below, once the edits are read
    with c[3]:
        if st.button('Reset to AHG 2021', width='stretch'):
            _set_spec(ahg2021.ahg_spec(n_mcs=1_000_000))
            st.rerun()
    if up is not None:
        try:
            loaded = specmod.from_json(up.getvalue().decode('utf-8'))
        except specmod.ModelSpecError as e:
            st.error('The file was not loaded:')
            for p in e.problems:
                st.markdown(f'- {p}')
        except UnicodeDecodeError:
            st.error('The file is not a UTF-8 text file.')
        else:
            _set_spec(loaded)
            st.session_state['lab_flash'] = f"Loaded “{loaded.get('name', 'model')}” ({up.name})."
            st.rerun()

    st.markdown('##### Global settings')
    c = st.columns(6)
    T_ref = _num_input(c[0], 'T_ref (years)', 0.01, 10_000.0, spec.get('T_ref'), 1.0, f'lab_Tref_{v}',
                       help='Reference period of the reliability (the analysis period is max(T_ref, T_basic)).')
    beta_t = _num_input(c[1], 'target β_t', 0.0, 8.0, spec.get('beta_target'), 0.1, f'lab_bt_{v}')
    n_aQ = c[2].number_input('n a_Q', 1, 200, int(min(max(int(spec.get('n_aQ') or 1), 1), 200)), 1, key=f'lab_naq_{v}',
                             help='Points of the a_Q range per failure mode.')
    n_aG = c[3].number_input('n a_G', 1, 200, int(min(max(int(spec.get('n_aG') or 1), 1), 200)), 1, key=f'lab_nag_{v}',
                             help='Points of the a_G range per failure mode.')
    seed = c[4].number_input('MCS seed', 0, 2 ** 31 - 1, int(min(max(int(spec.get('seed') or 0), 0), 2 ** 31 - 1)), 1,
                             key=f'lab_seed_{v}', help='Seed of the Monte-Carlo simulation of the characteristic load values.')
    mcs_options = [100_000, 1_000_000, 10_000_000]
    mcs_now = min(mcs_options, key=lambda n: abs(np.log(n) - np.log(max(int(spec.get('n_mcs') or 1), 1))))
    n_mcs = c[5].select_slider('MCS draws', mcs_options, mcs_now, key=f'lab_nmcs_{v}', format_func=lambda n: f'{n:,}',
                               help='Draws for the characteristic values of Θ_Q·Q. 10 M reproduces the AHG '
                                    'numbers exactly; 1 M differs in the third decimal and is ten times faster.')
    c = st.columns(4)
    tE_cov = _num_input(c[0], 'Θ_E CoV (load-effect model uncertainty)', 0.001, 2.0, spec['theta_E'].get('cov'), 0.01,
                        f'lab_tE_{v}')
    tE_mean = _num_input(c[1], 'Θ_E mean', 0.01, 10.0, spec['theta_E'].get('mean'), 0.01, f'lab_tEm_{v}')
    P_cov = _num_input(c[2], 'P CoV (other permanent load)', 0.001, 2.0, spec['P'].get('cov'), 0.01, f'lab_P_{v}')
    P_pinned = spec['P'].get('char') is not None
    if P_pinned:
        P_char = _num_input(c[3], 'P x_k (pinned)', 0.01, 10.0, spec['P'].get('char'), 0.01, f'lab_Pc_{v}',
                            help='This model pins p_k; edit the JSON to use a fractile instead.')
    else:
        P_frac = _num_input(c[3], 'P fractile for p_k', 0.001, 0.999, spec['P'].get('fractile'), 0.05, f'lab_Pf_{v}')

    st.markdown('##### Failure modes')
    modes_df = st.data_editor(modes_table(spec), num_rows='dynamic', key=f'lab_modes_{v}', width='stretch',
                              column_config={'include': st.column_config.CheckboxColumn(),
                                             'name': st.column_config.TextColumn(required=True)})
    st.markdown('##### Variable loads')
    loads_df = st.data_editor(loads_table(spec), num_rows='dynamic', key=f'lab_loads_{v}', width='stretch',
                              column_config={'include': st.column_config.CheckboxColumn(),
                                             'name': st.column_config.TextColumn(required=True),
                                             'q_k pinned': st.column_config.NumberColumn(
                                                 help='Leave empty to simulate q_k from the fractile.')})

    with C.guarded('reading the tables') as g:
        new = tables_to_spec(spec, modes_df, loads_df)
    if g.failed:
        return
    new.update(name=name, T_ref=float(T_ref), beta_target=float(beta_t), n_aQ=int(n_aQ), n_aG=int(n_aG),
               seed=int(seed), n_mcs=int(n_mcs))
    new['theta_E'] = dict(new['theta_E'], cov=float(tE_cov), mean=float(tE_mean))
    new['P'] = dict(new['P'], cov=float(P_cov), **({'char': float(P_char)} if P_pinned else {'fractile': float(P_frac)}))
    st.session_state['lab_spec'] = new       # the live model; the editors keep showing the base + their edits
    save_slot.download_button('Save model as JSON', specmod.to_json(new),
                              file_name=(re.sub(r'[^\w-]+', '_', name.strip()).strip('_')[:40] or 'calibration_model') + '.json',
                              mime='application/json', width='stretch')

    problems = specmod.validate_spec(new)
    if problems:
        st.error('The model has problems — fix them before running:')
        for p in problems:
            st.markdown(f'- {p}')
    else:
        diff = specmod.diff_specs(ahg2021.ahg_spec(n_mcs=int(n_mcs)), new)
        diff = [d for d in diff if d[0] not in ('name',)]
        if diff:
            with st.expander(f'{len(diff)} difference(s) from the AHG 2021 model'):
                st.dataframe(pd.DataFrame(diff, columns=['parameter', 'AHG 2021', 'this model']), hide_index=True,
                             width='stretch')
        else:
            st.success('Identical to the AHG 2021 model.')
        if st.checkbox('Build the model and show its random variables', True, key='lab_preview',
                       help='Untick while editing many cells at 10 M draws — every change rebuilds the model.'):
            with C.guarded('building the model') as g:
                with st.spinner('Characteristic values by Monte Carlo…'):
                    model = C.build_model(new)
                C.model_overview(model)


# ---- Status quo tab -------------------------------------------------------------------------

def _problems_banner(spec):
    problems = specmod.validate_spec(spec)
    if problems:
        st.error('The model is not valid (see the Model tab):')
        for p in problems[:5]:
            st.markdown(f'- {p}')
    return problems


def tab_status_quo():
    spec = _spec()
    if _problems_banner(spec):
        return
    c = st.columns(3)
    equation = c[0].selectbox('Design equation', list(C.EQUATION_LABELS), format_func=C.EQUATION_LABELS.get, key='sq_eq')
    method = c[1].radio('β method', ['form', 'sorm'], horizontal=True, key='sq_method', format_func=str.upper,
                        help='SORM corrects the first-order β for the curvature of the limit state (≈ +70 % time).')
    per_load = c[2].checkbox('γ_Q per load', key='sq_perload')
    pf = C.factor_controls(spec, 'sq', per_load=per_load, with_xi=equation != '8.12')
    n = C.situation_count(spec)
    st.caption(f'{n} design situations ≈ {C.estimate_seconds(n, method):.0f} s')
    if st.button('Run status quo', type='primary', key='sq_run'):
        with C.guarded('status-quo sweep') as g:
            with st.spinner('Building the model and running FORM…'):
                df = C.run_sweep(spec, pf, equation, method)
            st.session_state['sq_result'] = {'df': df, 'pf': pf, 'equation': equation, 'method': method,
                                             'spec': copy.deepcopy(spec)}
    out = st.session_state.get('sq_result')
    if not out:
        return
    if specmod.spec_signature(out['spec']) != specmod.spec_signature(spec):
        st.info('The model has changed since this result was computed — run again to update.')
    C.sweep_metrics(out['df'], out['spec']['beta_target'])
    C.sweep_plots(out['df'], out['spec']['beta_target'], 'sq')
    C.snapshot_panel(out['df'], out['pf'], out['equation'], out['method'], out['spec'], 'status quo', 'sq')
    st.download_button('Download sweep (CSV)', out['df'].to_csv(index=False), 'status_quo.csv', 'text/csv')


# ---- Calibration tab -------------------------------------------------------------------------

def tab_calibration():
    spec = _spec()
    if _problems_banner(spec):
        return
    loads = [l['name'] for l in spec['loads'] if l.get('include', True)]
    c = st.columns(3)
    equation = c[0].selectbox('Design equation', list(C.EQUATION_LABELS), format_func=C.EQUATION_LABELS.get, key='cal_eq')
    method = c[1].radio('β method', ['form', 'sorm'], horizontal=True, key='cal_method', format_func=str.upper)
    target_mode = c[2].radio('Target β_t', ['model value', 'status-quo weighted mean'], key='cal_target_mode',
                             help='"status-quo weighted mean" calibrates to the average reliability of the start values.')
    options = ['gamma_G', 'gamma_GP', 'gamma_Q'] + [f'gamma_Q:{n}' for n in loads] + ['xi']
    default = ['gamma_G', 'gamma_Q'] + (['xi'] if equation != '8.12' else [])
    free = st.multiselect('Free parameters', options, default, format_func=C.FREE_PARAMETER_LABELS.get,
                          key=f'cal_free_{equation}')
    st.markdown('Start values (also used for the parameters that are not free):')
    start = C.factor_controls(spec, 'cal', per_load=any(f.startswith('gamma_Q:') for f in free),
                              with_xi=equation != '8.12')
    with st.expander('Optimiser settings'):
        c = st.columns(3)
        ftol = c[0].select_slider('ftol', [1e-4, 1e-5, 1e-6, 1e-8], 1e-6, key='cal_ftol', format_func=lambda x: f'{x:.0e}')
        eps = c[1].select_slider('finite-difference step', [1e-3, 1e-4, 1e-5], 1e-4, key='cal_eps', format_func=lambda x: f'{x:.0e}')
        bounds_text = c[2].text_input('bounds (name=lo:hi, …)', '', key='cal_bounds',
                                      help='e.g. gamma_Q=1.2:2.5, xi=0.7:1.0 — defaults: γ 0–1.5 / 1–3.5, ξ 0.5–1.2')
    problems = []
    if 'gamma_Q' in free and any(f.startswith('gamma_Q:') for f in free):
        problems.append('Choose either a common γ_Q or per-load factors, not both.')
    if 'xi' in free and equation == '8.12':
        problems.append('ξ does not appear in eq. 8.12.')
    if not free:
        problems.append('Choose at least one free parameter.')
    bounds = None
    if bounds_text.strip():
        try:
            bounds = list(bounds_for(free, equation))
            for item in bounds_text.split(','):
                name, rng = item.split('=')
                lo, hi = (float(x) for x in rng.split(':'))
                if name.strip() not in free:
                    raise ValueError(f'{name.strip()} is not a free parameter')
                if not lo < hi:
                    raise ValueError(f'{name.strip()}: lower bound must be below the upper bound')
                bounds[free.index(name.strip())] = (lo, hi)
        except ValueError as e:
            problems.append(f'Bounds not understood ({e}); use name=lo:hi separated by commas.')
    for p in problems:
        st.warning(p)
    n = C.situation_count(spec)
    st.caption(f'{n} design situations per evaluation, typically 15–60 evaluations '
               f'(≈ {C.estimate_seconds(n, method, 30):.0f} s).')
    if st.button('Calibrate', type='primary', key='cal_run', disabled=bool(problems)):
        status = st.status('Calibrating…', expanded=True)
        with C.guarded('calibration') as g:
            model = C.build_model(spec)
            beta_t = spec['beta_target']
            if target_mode != 'model value':
                beta_t = modelmod.weighted_mean_beta(C.run_sweep(spec, start, equation, method))
            bar = status.progress(0.0)
            text = status.empty()

            def progress(k, x, val):
                bar.progress(min(k / 60, 0.98))
                text.markdown(f'evaluation {k}: ' + ', '.join(f'{C.gamma_label(n_)} = {v:.4f}' for n_, v in zip(free, x))
                              + f' → M = {val:.4e}')
            used_bounds = bounds_for(free, equation) if bounds is None else tuple(bounds)
            res = run_calibration(model, tuple(free), start=start, equation=equation, method=method,
                                   beta_target=beta_t, bounds=used_bounds, ftol=ftol, eps=eps, verbose=False,
                                   progress=progress)
            bar.progress(1.0)
            status.update(label=f'Calibration finished after {len(res.history)} evaluations', state='complete',
                          expanded=False)
            st.session_state['cal_result'] = {'res': res, 'equation': equation, 'method': method,
                                              'spec': copy.deepcopy(spec), 'free': tuple(free), 'bounds': used_bounds}
        if g.failed:
            status.update(label='Calibration failed', state='error', expanded=False)
    out = st.session_state.get('cal_result')
    if not out:
        return
    res = out['res']
    if specmod.spec_signature(out['spec']) != specmod.spec_signature(spec):
        st.info('The model has changed since this calibration — run again to update.')
    if not res.optimizer.success:
        st.warning(f'The optimiser stopped without declaring convergence: {res.optimizer.message}. '
                   'The best point found is shown.')
    at_bound = [n_ for n_, (lo, hi) in zip(res.free, out['bounds'])
                if abs(res.factors.get(n_) - lo) < 1e-6 or abs(res.factors.get(n_) - hi) < 1e-6]
    if at_bound:
        st.warning('At a bound: ' + ', '.join(C.gamma_label(n_) for n_ in at_bound) + ' — widen the bounds if that is not intended.')
    c = st.columns(len(res.free) + 2)
    for i, name in enumerate(res.free):
        c[i].metric(C.gamma_label(name), f'{res.factors.get(name):.3f}')
    c[-2].metric('weighted-mean β', f'{res.beta_after:.3f}', f'{res.beta_after - res.beta_before:+.3f} vs start')
    c[-1].metric('objective M', f'{res.objective:.4f}', f'{res.objective - res.history[0][1]:+.4f} vs start',
                 delta_color='inverse')
    st.caption(f'β_t = {res.beta_target:.3f}, eq. {out["equation"]}, {out["method"].upper()}, '
               f'{len(res.history)} evaluations, SLSQP: {res.optimizer.message}')
    C.factors_table(res.factors, res.free)
    C.sweep_plots(res.sweep_after, res.beta_target, 'cal', ref_df=res.sweep_before, labels=('calibrated', 'start values'))
    with st.expander('Optimiser history'):
        C.show(plots.plot_calibration_history(res))
        st.dataframe(pd.DataFrame([{**{C.gamma_label(n_): v for n_, v in zip(res.free, x)}, 'M': val}
                                   for x, val in res.history]).style.format('{:.5f}'), hide_index=True, width='stretch')
    C.snapshot_panel(res.sweep_after, res.factors, out['equation'], out['method'], out['spec'], 'calibration', 'cal',
                     extra={'free': res.free, 'objective': res.objective, 'beta_target': res.beta_target,
                            'before': res.sweep_before})
    st.download_button('Download calibrated sweep (CSV)', res.sweep_after.to_csv(index=False), 'calibrated.csv', 'text/csv')


# ---- Compare tab -------------------------------------------------------------------------------

def tab_compare():
    snaps = C.snapshots()
    if not snaps:
        st.info('No snapshots yet — save one under a status-quo or calibration result.')
        return
    st.dataframe(C.snapshot_table(), hide_index=True, width='stretch')
    names = [s['name'] for s in snaps]
    c = st.columns([2, 2, 1])
    a = c[0].selectbox('Show', names, key='cmp_a')
    b = c[1].selectbox('against (reference)', ['(none)'] + names, key='cmp_b')
    if c[2].button('Delete all snapshots', width='stretch'):
        st.session_state['snapshots'] = []
        st.rerun()
    sa = next(s for s in snaps if s['name'] == a)
    sb = next((s for s in snaps if s['name'] == b), None)
    if sb is not None and set(sa['df']['mode']) != set(sb['df']['mode']):
        st.warning('The two snapshots contain different failure modes — the overlays show the ones each has.')
    C.sweep_metrics(sa['df'], sa['beta_target'], None if sb is None else sb['df'])
    C.sweep_plots(sa['df'], sa['beta_target'], 'cmp', None if sb is None else sb['df'], labels=(a, b))
    cal = {s['name']: s for s in snaps if s['kind'] == 'calibration'}
    if len(cal) >= 2:
        st.markdown('##### Calibrated factors across snapshots')
        class _R:                       # minimal stand-in for CalibrationResult in the forest plot
            def __init__(self, s):
                self.free = s['free']
                self.factors = C.pf_from_dict(s['factors'])
        C.show(plots.plot_gamma_comparison({k: _R(s) for k, s in cal.items()}))


# ---- Sensitivity tab -----------------------------------------------------------------------------

def spec_paths(spec):
    """Numeric parameters of the spec that can be varied, as path strings."""
    paths = ['beta_target', 'T_ref', 'theta_E.cov', 'theta_E.mean', 'P.cov']
    for m in spec['modes']:
        if not m.get('include', True):
            continue
        n = m['name']
        paths += [f'modes[{n}].theta_R.mean', f'modes[{n}].theta_R.cov', f'modes[{n}].R.cov', f'modes[{n}].gamma_M',
                  f'modes[{n}].G.cov', f'modes[{n}].weight']
        if m['R'].get('char') is not None:
            paths.append(f'modes[{n}].R.char')
    for l in spec['loads']:
        if not l.get('include', True):
            continue
        n = l['name']
        paths += [f'loads[{n}].theta_Q.mean', f'loads[{n}].theta_Q.cov', f'loads[{n}].cov', f'loads[{n}].T_basic',
                  f'loads[{n}].fractile', f'loads[{n}].psi_0', f'loads[{n}].weight']
        if l.get('char') is not None:
            paths.append(f'loads[{n}].char')
    return paths


_TOKEN = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)(?:\[([^\]]+)\])?')


def get_by_path(spec, path):
    obj = spec
    for key, idx in _TOKEN.findall(path):
        obj = obj[key]
        if idx:
            obj = next(x for x in obj if x['name'] == idx)
    return obj


def set_by_path(spec, path, value):
    spec = copy.deepcopy(spec)
    parts = _TOKEN.findall(path)
    obj = spec
    for key, idx in parts[:-1]:
        obj = obj[key]
        if idx:
            obj = next(x for x in obj if x['name'] == idx)
    obj[parts[-1][0]] = value
    return spec


def tab_sensitivity():
    spec = _spec()
    if _problems_banner(spec):
        return
    st.markdown('Vary one parameter of the model and watch the status-quo reliability — or the calibrated '
                'factors — respond. The quickest way to find out which assumptions matter.')
    paths = spec_paths(spec)
    c = st.columns([3, 1, 1, 1])
    path = c[0].selectbox('Parameter', paths, index=paths.index('loads[wind].theta_Q.cov') if 'loads[wind].theta_Q.cov' in paths else 0,
                          key='sens_path')
    current = float(get_by_path(spec, path))
    lo0, hi0 = round(current * 0.5, 4), round(current * 1.5, 4)
    if path.endswith(('fractile', 'psi_0')):
        lo0, hi0 = round(max(current - 0.1, 0.01), 4), round(min(current + 0.1, 0.999), 4)
    lo = c[1].number_input('from', value=lo0, key=f'sens_lo_{path}', format='%.4f')
    hi = c[2].number_input('to', value=hi0, key=f'sens_hi_{path}', format='%.4f')
    npts = c[3].number_input('points', 2, 15, 5, 1, key='sens_n')
    c = st.columns(3)
    what = c[0].radio('Compute', ['status-quo weighted-mean β', 'calibrated γ_G and γ_Q'], key='sens_what')
    equation = c[1].selectbox('Design equation', list(C.EQUATION_LABELS), format_func=C.EQUATION_LABELS.get, key='sens_eq')
    grid = c[2].radio('Grid', ['model grid', 'quick (1 × 5)'], horizontal=True, key='sens_grid')
    pf = C.factor_controls(spec, 'sens', with_xi=equation != '8.12')
    if not lo < hi:
        st.warning('"from" must be below "to".')
        return
    spec_run = dict(spec) if grid == 'model grid' else dict(spec, n_aG=1, n_aQ=5)
    n = C.situation_count(spec_run)
    evals = 1 if what.startswith('status') else 30
    st.caption(f'{npts} model variants × {n} situations' + ('' if evals == 1 else ' × ~30 evaluations')
               + f' ≈ {C.estimate_seconds(n, "form", evals) * npts:.0f} s')
    if st.button('Run sensitivity study', type='primary', key='sens_run'):
        rows = []
        bar = st.progress(0.0)
        with C.guarded('sensitivity study') as g:
            for i, value in enumerate(np.linspace(lo, hi, int(npts))):
                sp = set_by_path(spec_run, path, float(value))
                problems = specmod.validate_spec(sp)
                if problems:
                    raise ValueError(f'{path} = {value:.4g} makes the model invalid: {problems[0]}')
                if what.startswith('status'):
                    df = C.run_sweep(sp, pf, equation, 'form')
                    rows.append({path: value, 'weighted-mean β': modelmod.weighted_mean_beta(df),
                                 'min β': df.beta.min(), 'max β': df.beta.max(),
                                 'objective': objective(df, sp['beta_target'])})
                else:
                    model = C.build_model(sp)
                    res = run_calibration(model, ('gamma_G', 'gamma_Q'), start=pf, equation=equation,
                                           beta_target=sp['beta_target'], verbose=False, ftol=1e-5, eps=1e-4)
                    rows.append({path: value, 'γ_G': res.factors.gamma_G, 'γ_Q': res.factors.get('gamma_Q'),
                                 'weighted-mean β': res.beta_after, 'objective': res.objective})
                bar.progress((i + 1) / int(npts))
        bar.empty()
        if g.failed or not rows:
            return
        st.session_state['sens_result'] = (path, pd.DataFrame(rows))
    out = st.session_state.get('sens_result')
    if not out:
        return
    path, df = out
    ycols = [c_ for c_ in df.columns if c_ not in (path, 'objective')]
    fig = Figure(figsize=(8, 3.4))
    ax = fig.subplots()
    for c_ in ycols:
        ax.plot(df[path], df[c_], 'o-', label=c_)
    try:
        ax.axvline(float(get_by_path(spec, path)), color='grey', ls=':', label='current value')
    except (KeyError, StopIteration):       # the parameter no longer exists in the edited model
        pass
    ax.set_xlabel(path)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    C.show(fig)
    st.dataframe(df.style.format('{:.4f}'), hide_index=True, width='stretch')
    st.download_button('Download (CSV)', df.to_csv(index=False), 'sensitivity.csv', 'text/csv')


# ---- Verification tab ------------------------------------------------------------------------------

def tab_verification():
    spec = _spec()
    if _problems_banner(spec):
        return
    st.markdown('FORM against SORM and importance sampling for a sample of the design situations.')
    c = st.columns(3)
    step = c[0].select_slider('every k-th situation', [1, 3, 9, 27, 81], 9, key='ver_step')
    equation = c[1].selectbox('Design equation', list(C.EQUATION_LABELS), format_func=C.EQUATION_LABELS.get, key='ver_eq')
    n_is = c[2].select_slider('importance-sampling samples', [5_000, 20_000, 100_000], 20_000, key='ver_nis')
    pf = C.factor_controls(spec, 'ver', with_xi=equation != '8.12')
    if st.button('Verify', type='primary', key='ver_run'):
        with C.guarded('verification') as g:
            model = C.build_model(spec)
            sits = model.design_situations()[::step]
            rows = []
            bar = st.progress(0.0, text=f'{len(sits)} situations')
            for i, s in enumerate(sits):
                z = s.design_parameter(pf, equation)
                ls = s.limit_state(z)
                r = reliability.form(ls)
                b_s, kappa = reliability.sorm(ls, r)
                isr = reliability.importance_sampling(ls, r, n=n_is, seed=i)
                rows.append({'mode': s.mode.name, 'load': s.load.name, 'a_G': s.a_G, 'a_Q': s.a_Q, 'z': z,
                             'β FORM': r.beta, 'β SORM': b_s, 'β IS': isr.beta, 'se IS': isr.se_beta,
                             'converged': r.converged, 'κ min': kappa.min(),
                             'dominant': s.variable_names[int(np.argmax(np.abs(r.alpha)))]})
                if i % 10 == 0:
                    bar.progress((i + 1) / len(sits))
            bar.empty()
            st.session_state['ver_result'] = pd.DataFrame(rows)
    ver = st.session_state.get('ver_result')
    if ver is None:
        return
    d_f, d_s = ver['β FORM'] - ver['β IS'], ver['β SORM'] - ver['β IS']
    c = st.columns(4)
    c[0].metric('FORM − IS, mean', f'{d_f.mean():+.4f}')
    c[1].metric('FORM − IS, max |·|', f'{d_f.abs().max():.4f}')
    c[2].metric('SORM − IS, mean', f'{d_s.mean():+.4f}')
    c[3].metric('within 2 s.e. of IS', f'FORM {np.mean(d_f.abs() < 2 * ver["se IS"]):.0%} · SORM {np.mean(d_s.abs() < 2 * ver["se IS"]):.0%}')
    if not ver['converged'].all():
        st.warning(f'{int((~ver["converged"]).sum())} situations: FORM did not converge.')
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
    st.dataframe(ver.groupby(['load', 'dominant'])[['β FORM', 'β SORM', 'β IS']].mean().assign(
        **{'FORM − IS': lambda t: t['β FORM'] - t['β IS'], 'SORM − IS': lambda t: t['β SORM'] - t['β IS']})
        .style.format('{:.3f}'), width='stretch')
    st.download_button('Download (CSV)', ver.to_csv(index=False), 'verification.csv', 'text/csv')


# ---- entry ------------------------------------------------------------------------------------------

def render():
    st.title('JCSS CodeCal 2.0 — lab mode')
    st.caption('Edit the probabilistic model, evaluate the status quo, calibrate, compare, test the assumptions.')
    tabs = st.tabs(['Model', 'Status quo', 'Calibration', 'Compare', 'Sensitivity', 'Verification'])
    with tabs[0]:
        tab_model()
    with tabs[1]:
        tab_status_quo()
    with tabs[2]:
        tab_calibration()
    with tabs[3]:
        tab_compare()
    with tabs[4]:
        tab_sensitivity()
    with tabs[5]:
        tab_verification()
