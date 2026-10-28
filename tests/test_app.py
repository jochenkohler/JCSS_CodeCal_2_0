"""
Headless tests of the Streamlit app (streamlit.testing.v1.AppTest): both
modes render, every run button produces a result, validation gates the
runs, and the Model tab's table ↔ spec round trip is exact.
"""

import copy
import os
import sys

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault('MPLBACKEND', 'Agg')          # no GUI backend in test threads (macOS)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

pytest.importorskip('streamlit')
from streamlit.testing.v1 import AppTest           # noqa: E402

import codecal as cc                                # noqa: E402
from codecal.gui import lab                         # noqa: E402

APP = os.path.join(ROOT, 'app.py')


def _app():
    at = AppTest.from_file(APP, default_timeout=600)
    at.run()
    return at


def _no_exception(at, where):
    assert not at.exception, f'{where}: ' + '\n'.join(str(e) for e in at.exception)


# ---- learning mode ------------------------------------------------------------------

@pytest.fixture(scope='module')
def learning():
    at = _app()
    _no_exception(at, 'learning start')
    return at


def test_learning_mode_opens_on_step_1(learning):
    assert learning.title[0].value.endswith('learning mode')
    assert learning.sidebar.radio('learning_step').value.startswith('1')
    assert learning.metric[0].label.startswith('x_k')


def test_learning_steps_render_and_run(learning):
    at = learning
    steps = at.sidebar.radio('learning_step').options
    assert len(steps) == 6

    at.sidebar.radio('learning_step').set_value(steps[1]).run()
    _no_exception(at, 'step 2')
    at.radio('l2_kind').set_value(at.radio('l2_kind').options[1]).run()
    _no_exception(at, 'step 2 product')
    beta_form = float(at.metric[0].value)
    assert 3.0 < beta_form < 3.2                       # c − Θ·Q example: FORM 3.105

    at.sidebar.radio('learning_step').set_value(steps[2]).run()
    _no_exception(at, 'step 3')
    at.selectbox('l3_eq').set_value('8.13').run()
    _no_exception(at, 'step 3 eq 8.13')

    at.sidebar.radio('learning_step').set_value(steps[3]).run()
    at.radio('l4_grid').set_value('quick (1 × 5)').run()
    at.button('l4_run').click().run()
    _no_exception(at, 'step 4 sweep')
    wm = float(at.metric[0].value)
    assert 3.6 < wm < 3.9
    for view in at.radio('l4_view').options:
        at.radio('l4_view').set_value(view).run()
        _no_exception(at, f'step 4 view {view}')

    at.sidebar.radio('learning_step').set_value(steps[4]).run()
    at.radio('l5_grid').set_value('quick (1 × 5)').run()
    at.button('l5_run').click().run()
    _no_exception(at, 'step 5 calibrate')
    gG, gQ = float(at.metric[0].value), float(at.metric[1].value)
    assert 1.15 < gG < 1.30 and 1.65 < gQ < 1.85     # full grid: 1.215 / 1.750
    at.multiselect('l5_free_8.12').set_value(['gamma_Q', 'gamma_Q:wind']).run()
    assert at.button('l5_run').disabled
    assert any('either a common' in w.value for w in at.warning)

    at.sidebar.radio('learning_step').set_value(steps[5]).run()
    at.select_slider('l6_step').set_value(81).run()
    at.button('l6_run').click().run()
    _no_exception(at, 'step 6 verify')
    form_minus_is = float(at.metric[0].value)
    assert 0.0 < form_minus_is < 0.1                  # FORM is optimistic by a few hundredths


# ---- lab mode --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def lab_app():
    at = _app()
    at.sidebar.radio('app_mode').set_value('Lab mode').run()
    _no_exception(at, 'lab start')
    at.number_input('lab_naq_0').set_value(3).run()       # small grid: 9 × 3 × 1 × 3 = 81 situations
    at.number_input('lab_nag_0').set_value(1).run()
    _no_exception(at, 'lab small grid')
    return at


def test_lab_model_tab(lab_app):
    at = lab_app
    assert [t.label for t in at.tabs] == ['Model', 'Status quo', 'Calibration', 'Compare', 'Sensitivity', 'Verification']
    assert any(e.label.startswith('2 difference') for e in at.expander)
    spec = at.session_state['lab_spec']
    assert spec['n_aQ'] == 3 and spec['n_aG'] == 1
    assert cc.validate_spec(spec) == []


def test_lab_status_quo_and_snapshot(lab_app):
    at = lab_app
    at.button('sq_run').click().run()
    _no_exception(at, 'status quo')
    labels = [m.label for m in at.metric]
    wm = float(at.metric[labels.index('weighted-mean β')].value)
    assert 3.6 < wm < 3.9
    at.text_input('sq_snapname').set_value('status quo').run()
    at.button('sq_snapbtn').click().run()
    assert len(at.session_state['snapshots']) == 1
    at.radio('sq_method').set_value('sorm').run()
    at.checkbox('sq_perload').check().run()
    at.button('sq_run').click().run()
    _no_exception(at, 'status quo SORM per load')


def test_lab_calibration_gating_and_run(lab_app):
    at = lab_app
    at.text_input('cal_bounds').set_value('nonsense').run()
    assert at.button('cal_run').disabled
    assert any('Bounds not understood' in w.value for w in at.warning)
    at.text_input('cal_bounds').set_value('gamma_Q=1.2:2.5').run()
    assert not at.button('cal_run').disabled
    at.button('cal_run').click().run()
    _no_exception(at, 'calibration')
    labels = [m.label for m in at.metric]
    gG = float(at.metric[labels.index('γ_G')].value)
    gQ = float(at.metric[labels.index('γ_Q')].value)
    assert 1.15 < gG < 1.30 and 1.65 < gQ < 1.85
    at.text_input('cal_snapname').set_value('calibrated').run()
    at.button('cal_snapbtn').click().run()
    assert len(at.session_state['snapshots']) == 2


def test_lab_compare_tab(lab_app):
    at = lab_app
    at.selectbox('cmp_a').set_value('calibrated').run()
    at.selectbox('cmp_b').set_value('status quo').run()
    _no_exception(at, 'compare')
    deltas = [m.delta for m in at.metric if m.label == 'weighted-mean β']   # status quo, calibration, compare tabs
    assert any('vs reference' in d for d in deltas)


def test_lab_sensitivity_tab(lab_app):
    at = lab_app
    at.radio('sens_grid').set_value('quick (1 × 5)').run()
    at.number_input('sens_n').set_value(2).run()
    at.button('sens_run').click().run()
    _no_exception(at, 'sensitivity')
    path, df = at.session_state['sens_result']
    assert path == 'loads[wind].theta_Q.cov' and len(df) == 2
    assert df['weighted-mean β'].iloc[0] > df['weighted-mean β'].iloc[1]   # more wind uncertainty, less reliability


def test_lab_verification_tab(lab_app):
    at = lab_app
    at.select_slider('ver_step').set_value(27).run()
    at.button('ver_run').click().run()
    _no_exception(at, 'verification')
    ver = at.session_state['ver_result']
    assert len(ver) == 3 and ver['converged'].all()
    assert 0 < (ver['β FORM'] - ver['β IS']).mean() < 0.1


def test_lab_invalid_model_blocks_the_runs(lab_app):
    at = lab_app
    bad = copy.deepcopy(at.session_state['lab_base'])
    bad['modes'][0]['weight'] = -1
    bad['loads'][0]['psi_0'] = 2
    at.session_state['lab_base'] = bad
    at.session_state['lab_spec_version'] = 9
    at.run()
    _no_exception(at, 'invalid model')
    assert any(e.value.startswith('The model has problems') for e in at.error)
    problems = [m.value for m in at.markdown if m.value.startswith('- ')]
    assert any('weight must be' in p for p in problems) and any('ψ₀' in p for p in problems)
    assert 'sq_run' not in [b.key for b in at.button]             # run buttons are not offered
    # stale-result notice and reset
    at.button[[b.label for b in at.button].index('Reset to AHG 2021')].click().run()
    _no_exception(at, 'reset')
    assert any(s.value.startswith('Identical to the AHG') for s in at.success)
    assert any('changed since' in i.value for i in at.info)


# ---- the table ↔ spec round trip of the Model tab ---------------------------------------

def test_tables_round_trip_and_edits():
    spec = cc.ahg_spec()
    modes, loads = lab.modes_table(spec), lab.loads_table(spec)
    assert cc.diff_specs(spec, lab.tables_to_spec(spec, modes, loads)) == []

    # edit a cell, pin x_k instead of the fractile, untick a mode, rename a load
    modes.loc[1, 'Θ_R CoV'] = 0.2
    modes.loc[2, 'R fractile'] = np.nan
    modes.loc[2, 'R x_k'] = 0.9
    modes.loc[3, 'include'] = False
    loads.loc[0, 'name'] = 'storm'
    new = lab.tables_to_spec(spec, modes, loads)
    assert new['modes'][1]['theta_R']['cov'] == 0.2
    assert 'fractile' not in new['modes'][2]['R'] and new['modes'][2]['R']['char'] == 0.9
    assert new['modes'][3]['include'] is False
    assert new['loads'][0]['name'] == 'storm' and new['loads'][0]['theta_Q'] == spec['loads'][0]['theta_Q']
    assert cc.validate_spec(new) == []

    # delete a row and add one (as the dynamic editor does: index kept / appended)
    modes2 = lab.modes_table(spec).drop(index=0)
    added = pd.DataFrame([{'include': True, 'name': 'Masonry shear', 'Θ_R mean': 1.0, 'Θ_R CoV': 0.2, 'R CoV': 0.2,
                           'R x_k': np.nan, 'R fractile': 0.05, 'γ_M': 1.5, 'G CoV': 0.07, 'G x_k': np.nan,
                           'G fractile': 0.5, 'a_Q min': 0.1, 'a_Q max': 0.7, 'a_G min': 0.6, 'a_G max': 1.0,
                           'weight': 0.05}], index=[9])
    modes2 = pd.concat([modes2, added])
    new2 = lab.tables_to_spec(spec, modes2, loads)
    assert [m['name'] for m in new2['modes']][:2] == ['Concrete compression', 'Concrete bending']
    assert new2['modes'][-1]['name'] == 'Masonry shear'
    assert new2['modes'][-1]['R']['name'] == 'Resistance Masonry shear'
    assert cc.validate_spec(new2) == []
    model = cc.build_model({**new2, 'n_mcs': 20_000})
    assert len(model.failure_modes) == 9

    # both x_k and fractile given → reported, not silently resolved
    modes3 = lab.modes_table(spec)
    modes3.loc[0, 'R fractile'] = 0.05                 # steel R has x_k pinned already
    problems = cc.validate_spec(lab.tables_to_spec(spec, modes3, loads))
    assert any('either a pinned' in p for p in problems)


def test_spec_paths_get_and_set():
    spec = cc.ahg_spec()
    paths = lab.spec_paths(spec)
    assert 'loads[wind].theta_Q.cov' in paths and 'modes[Steel bending].R.char' in paths
    assert 'modes[Concrete bending].R.char' not in paths      # that one uses a fractile
    for p in paths:
        v = lab.get_by_path(spec, p)
        assert isinstance(v, (int, float))
        new = lab.set_by_path(spec, p, v * 1.1)
        assert lab.get_by_path(new, p) == pytest.approx(v * 1.1)
        assert lab.get_by_path(spec, p) == v                 # the original is untouched
