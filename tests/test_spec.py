import copy
import json
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import codecal as cc                        # noqa: E402


def test_ahg_spec_is_valid_and_json_round_trips():
    spec = cc.ahg_spec()
    assert cc.validate_spec(spec) == []
    text = cc.to_json(spec)
    back = cc.from_json(text)
    assert back == json.loads(text)
    assert cc.diff_specs(spec, back) == []
    assert cc.spec_signature(spec) == cc.spec_signature(back)


def test_build_model_from_spec_matches_ahg_model():
    a = cc.ahg_model(n_mcs=100_000)
    b = cc.build_model(cc.ahg_spec(n_mcs=100_000))
    assert [m.name for m in a.failure_modes] == [m.name for m in b.failure_modes]
    assert a.loads[0].Q.char == b.loads[0].Q.char
    assert a.failure_modes[0].R.name == 'Steel yield strength'
    assert a.loads[2].Q.note.startswith('q_k pinned')


@pytest.mark.parametrize('edit,fragment', [
    (lambda s: s.__setitem__('T_ref', -5), 'T_ref'),
    (lambda s: s.__setitem__('n_aQ', 0), 'n_aQ'),
    (lambda s: s['modes'][0].__setitem__('a_Q', [0.8, 0.3]), 'a_Q range'),
    (lambda s: s['modes'][0]['R'].__setitem__('cov', -0.1), 'CoV'),
    (lambda s: s['modes'][0]['R'].update({'fractile': 0.05}), 'either a pinned'),
    (lambda s: s['modes'][1].__setitem__('name', 'Steel bending'), 'duplicate'),
    (lambda s: s['loads'][0].__setitem__('psi_0', 1.5), 'ψ₀'),
    (lambda s: s['loads'][0].__setitem__('T_basic', 0), 'T_basic'),
    (lambda s: [m.__setitem__('include', False) for m in s['modes']], 'at least one failure mode'),
    (lambda s: [l.__setitem__('weight', 0.0) for l in s['loads']], 'at least one load'),
    (lambda s: s['theta_E'].__setitem__('family', 'weibull'), 'family'),
])
def test_validation_catches_bad_input(edit, fragment):
    spec = cc.ahg_spec()
    edit(spec)
    problems = cc.validate_spec(spec)
    assert any(fragment in p for p in problems), problems
    with pytest.raises(cc.ModelSpecError) as e:
        cc.build_model(spec)
    assert fragment in str(e.value)


def test_validation_collects_all_problems_at_once():
    spec = cc.ahg_spec()
    spec['T_ref'] = 0
    spec['modes'][0]['gamma_M'] = -1
    spec['loads'][1]['fractile'] = 2
    assert len(cc.validate_spec(spec)) == 3


def test_excluded_modes_and_loads_are_left_out():
    spec = cc.ahg_spec(n_mcs=50_000)
    spec['modes'][0]['include'] = False
    spec['loads'][1]['include'] = False
    model = cc.build_model(spec)
    assert len(model.failure_modes) == 8 and len(model.loads) == 2
    df = cc.evaluate(cc.build_model({**spec, 'n_aQ': 2, 'n_aG': 1}), cc.START_VALUES)
    assert set(df['mode']) == {m['name'] for m in spec['modes'][1:]}
    assert df['weight'].sum() == pytest.approx(1.0)
    assert df['converged'].all()


def test_from_json_reports_syntax_and_content_errors():
    with pytest.raises(cc.ModelSpecError) as e:
        cc.from_json('{"T_ref": 50,')
    assert 'not valid JSON' in str(e.value)
    with pytest.raises(cc.ModelSpecError):
        cc.from_json('{"T_ref": 50}')


def test_diff_specs_names_the_changed_entry():
    a, b = cc.ahg_spec(), cc.ahg_spec()
    b['modes'][2]['weight'] = 0.3
    b['loads'][0]['theta_Q']['cov'] = 0.15
    d = cc.diff_specs(a, b)
    paths = [p for p, _, _ in d]
    assert paths == ['loads[wind].theta_Q.cov', 'modes[Concrete bending].weight']


def test_calibrate_progress_callback_and_empty_free():
    model = cc.build_model(cc.ahg_spec(n_mcs=50_000, n_aQ=2, n_aG=1))
    calls = []
    res = cc.calibrate(model, ('gamma_Q',), verbose=False, ftol=1e-3, eps=1e-3,
                       progress=lambda k, x, v: calls.append((k, float(v))))
    assert len(calls) == len(res.history) > 0
    with pytest.raises(ValueError):
        cc.calibrate(model, (), verbose=False)


def test_cli_runs_a_model_file(tmp_path):
    """run_calibration.py --model: a model saved by the lab mode is calibrated from the command line."""
    import run_calibration
    spec = cc.ahg_spec(n_mcs=50_000, n_aQ=2, n_aG=1)
    spec['modes'][0]['theta_R']['cov'] = 0.08
    path = tmp_path / 'm.json'
    path.write_text(cc.to_json(spec), encoding='utf-8')
    out = tmp_path / 'out'
    run_calibration.main(['--model', str(path), '--free', 'gamma_Q', '--out', str(out)])
    assert (out / 'summary.txt').exists() and (out / 'calibrated.csv').exists()
    bad = tmp_path / 'bad.json'
    bad.write_text('{"T_ref": 50}', encoding='utf-8')
    with pytest.raises(cc.ModelSpecError):
        run_calibration.main(['--model', str(bad), '--out', str(out)])
