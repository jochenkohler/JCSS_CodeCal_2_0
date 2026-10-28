"""
Model specification: the calibration model as plain data.

A *spec* is a dictionary (JSON-serialisable) that holds every number of a
calibration model — the probabilistic model, the failure modes, the loads,
the sweep settings. `build_model(spec)` turns it into a `CalibrationModel`,
`validate_spec(spec)` lists everything that is wrong with it before any
computation starts, and `ahg2021.ahg_spec()` is the AHG model of 2021 in
this form.

Why a dictionary and not objects? So that a model can be edited in a
table, saved to a file, loaded again, diffed against the default and used
as a cache key — the things a GUI and a reproducible study need.

Shape
-----
    rv      {'family': 'lognormal'|'normal'|'gumbel', 'mean': 1.0, 'cov': 0.05,
             'char': 0.83}            — or 'fractile': 0.05 instead of 'char';
             optional 'name' and 'note'
    mode    {'name', 'theta_R': rv, 'R': rv, 'gamma_M', 'G': rv,
             'a_Q': [min, max], 'a_G': [min, max], 'weight', 'include': True}
    load    {'name', 'theta_Q': rv, 'mean', 'cov', 'T_input', 'T_basic',
             'fractile', 'char': None | value, 'psi_0', 'weight', 'include': True}
    spec    {'name', 'T_ref', 'n_aQ', 'n_aG', 'beta_target', 'seed', 'n_mcs',
             'theta_E': rv, 'P': rv, 'modes': [mode, ...], 'loads': [load, ...]}
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np

from .distributions import FAMILIES
from .model import CalibrationModel, FailureMode, VariableLoad
from .variables import rv, time_variant_load

class ModelSpecError(ValueError):
    """The specification is inconsistent; `.problems` lists every issue."""

    def __init__(self, problems):
        self.problems = list(problems)
        super().__init__("Invalid model specification:\n  - " + "\n  - ".join(self.problems))


# ---- validation ---------------------------------------------------------------------

def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _check_rv(d, label, problems):
    if not isinstance(d, dict):
        problems.append(f"{label}: expected a distribution {{family, mean, cov, char|fractile}}")
        return
    fam = str(d.get('family', '')).lower()
    if fam not in FAMILIES:
        problems.append(f"{label}: unknown distribution family {d.get('family')!r} (normal, lognormal, gumbel)")
    mean, cov = _num(d.get('mean')), _num(d.get('cov'))
    if mean is None or mean <= 0:
        problems.append(f"{label}: mean must be a positive number (got {d.get('mean')!r})")
    if cov is None or not (0 < cov <= 2):
        problems.append(f"{label}: CoV must be between 0 and 2 (got {d.get('cov')!r})")
    has_char = d.get('char') is not None
    has_frac = d.get('fractile') is not None
    if has_char == has_frac:
        problems.append(f"{label}: give either a pinned characteristic value or a fractile, not both/neither")
    if has_char and (_num(d['char']) is None or _num(d['char']) <= 0):
        problems.append(f"{label}: characteristic value must be positive (got {d['char']!r})")
    if has_frac and (_num(d['fractile']) is None or not (0 < _num(d['fractile']) < 1)):
        problems.append(f"{label}: fractile must lie between 0 and 1 (got {d['fractile']!r})")


def _check_range(r, label, problems):
    try:
        lo, hi = float(r[0]), float(r[1])
    except (TypeError, ValueError, IndexError):
        problems.append(f"{label}: expected [min, max]")
        return
    if not (0 <= lo <= hi <= 1):
        problems.append(f"{label}: need 0 ≤ min ≤ max ≤ 1 (got {lo:g}, {hi:g})")


def validate_spec(spec) -> list:
    """Every problem with the specification, as readable sentences. An
    empty list means `build_model` will succeed."""
    p = []
    if not isinstance(spec, dict):
        return ["the specification must be a dictionary"]
    for key, lo, hi in (('T_ref', 1e-6, 1e4), ('beta_target', 0.0, 8.0)):
        v = _num(spec.get(key))
        if v is None or not (lo <= v <= hi):
            p.append(f"{key} must be a number between {lo:g} and {hi:g} (got {spec.get(key)!r})")
    for key, lo in (('n_aQ', 1), ('n_aG', 1), ('n_mcs', 1000)):
        v = spec.get(key)
        if not isinstance(v, (int, np.integer)) or v < lo:
            p.append(f"{key} must be an integer ≥ {lo} (got {v!r})")
    if spec.get('seed') is not None and not isinstance(spec.get('seed'), (int, np.integer)):
        p.append(f"seed must be an integer or null (got {spec.get('seed')!r})")
    _check_rv(spec.get('theta_E'), 'load-effect model uncertainty Θ_E', p)
    _check_rv(spec.get('P'), 'permanent load P', p)

    modes = spec.get('modes') or []
    names = set()
    active = 0
    for i, m in enumerate(modes):
        label = f"failure mode {i + 1} ({m.get('name', '?')})"
        name = str(m.get('name', '')).strip()
        if not name:
            p.append(f"{label}: name is empty")
        elif name in names:
            p.append(f"{label}: duplicate name")
        names.add(name)
        _check_rv(m.get('theta_R'), f"{label}: Θ_R", p)
        _check_rv(m.get('R'), f"{label}: R", p)
        _check_rv(m.get('G'), f"{label}: self-weight G", p)
        gm = _num(m.get('gamma_M'))
        if gm is None or gm <= 0:
            p.append(f"{label}: γ_M must be positive")
        _check_range(m.get('a_Q'), f"{label}: a_Q range", p)
        _check_range(m.get('a_G'), f"{label}: a_G range", p)
        w = _num(m.get('weight'))
        if w is None or w < 0:
            p.append(f"{label}: weight must be ≥ 0")
        elif m.get('include', True) and w > 0:
            active += 1
    if active == 0:
        p.append("at least one failure mode with a positive weight must be included")

    loads = spec.get('loads') or []
    names = set()
    active = 0
    for i, l in enumerate(loads):
        label = f"load {i + 1} ({l.get('name', '?')})"
        name = str(l.get('name', '')).strip()
        if not name:
            p.append(f"{label}: name is empty")
        elif name in names:
            p.append(f"{label}: duplicate name")
        names.add(name)
        _check_rv(l.get('theta_Q'), f"{label}: Θ_Q", p)
        for key in ('mean', 'cov', 'T_input', 'T_basic'):
            v = _num(l.get(key))
            if v is None or v <= 0:
                p.append(f"{label}: {key} must be positive (got {l.get(key)!r})")
        cov = _num(l.get('cov'))
        if cov is not None and cov > 2:
            p.append(f"{label}: CoV {cov:g} is implausibly large")
        fr = _num(l.get('fractile'))
        if fr is None or not (0 < fr < 1):
            p.append(f"{label}: fractile must lie between 0 and 1")
        if l.get('char') is not None and (_num(l['char']) is None or _num(l['char']) <= 0):
            p.append(f"{label}: pinned characteristic value must be positive")
        psi = _num(l.get('psi_0'))
        if psi is None or not (0 <= psi <= 1):
            p.append(f"{label}: ψ₀ must lie between 0 and 1")
        w = _num(l.get('weight'))
        if w is None or w < 0:
            p.append(f"{label}: weight must be ≥ 0")
        elif l.get('include', True) and w > 0:
            active += 1
    if active == 0:
        p.append("at least one load with a positive weight must be included")
    return p


# ---- building ------------------------------------------------------------------------

def _build_rv(d, name, symbol=''):
    kw = {'char': float(d['char'])} if d.get('char') is not None else {'fractile': float(d['fractile'])}
    return rv(d.get('name') or name, d['family'], float(d['mean']), float(d['cov']),
              symbol=symbol, note=d.get('note', ''), **kw)


def build_model(spec, strict=True) -> CalibrationModel:
    """CalibrationModel from a spec. Raises ModelSpecError listing every
    problem when the spec is invalid. Excluded modes/loads ('include':
    False) are left out of the model."""
    problems = validate_spec(spec)
    if problems:
        raise ModelSpecError(problems)
    spec = copy.deepcopy(spec)

    theta_E = _build_rv(spec['theta_E'], 'Load-effect model uncertainty', 'θ_E')
    P = _build_rv(spec['P'], 'Permanent load (other than self-weight)', 'P')

    modes = []
    for m in spec['modes']:
        if not m.get('include', True):
            continue
        modes.append(FailureMode(
            m['name'],
            _build_rv(m['theta_R'], f"Θ_R {m['name']}", 'θ_R'),
            _build_rv(m['R'], f"R {m['name']}", 'R'),
            float(m['gamma_M']),
            _build_rv(m['G'], f"Self-weight {m['name']}", 'G'),
            tuple(float(x) for x in m['a_Q']), tuple(float(x) for x in m['a_G']),
            float(m['weight'])))

    seed = spec.get('seed')
    rng = np.random if seed is None else np.random.RandomState(int(seed))
    loads = []
    for l in spec['loads']:
        if not l.get('include', True):
            continue
        theta_Q = _build_rv(l['theta_Q'], f"{l['name'].capitalize()} model uncertainty", 'θ_Q')
        Q, _ = time_variant_load(
            f"{l['name'].capitalize()}, {float(l['T_input']):g}-year maximum",
            float(l['mean']), float(l['cov']), T_input=float(l['T_input']),
            T_basic=float(l['T_basic']), T_ref=float(spec['T_ref']), theta_Q=theta_Q,
            fractile=float(l['fractile']),
            char=None if l.get('char') is None else float(l['char']),
            n_mcs=int(spec['n_mcs']), rng=rng, symbol='Q', note=l.get('note', ''))
        loads.append(VariableLoad(l['name'], theta_Q, Q, float(l['psi_0']), float(l['weight']),
                                  T_basic=float(l['T_basic'])))

    return CalibrationModel(
        failure_modes=modes, loads=loads, theta_E=theta_E, P=P,
        T_ref=float(spec['T_ref']), n_aQ=int(spec['n_aQ']), n_aG=int(spec['n_aG']),
        beta_target=float(spec['beta_target']), description=str(spec.get('name', '')))


# ---- JSON I/O -------------------------------------------------------------------------

def to_json(spec, indent=1) -> str:
    return json.dumps(spec, indent=indent, ensure_ascii=False)


def from_json(text) -> dict:
    """Parse a spec from JSON text; raises ModelSpecError if it is not
    valid JSON or not a valid specification."""
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as e:
        raise ModelSpecError([f"not valid JSON: {e.msg} (line {e.lineno})"]) from None
    problems = validate_spec(spec)
    if problems:
        raise ModelSpecError(problems)
    return spec


def spec_signature(spec) -> str:
    """Canonical string of a spec, usable as a cache key."""
    return json.dumps(spec, sort_keys=True, default=float)


def diff_specs(a, b, path=''):
    """List of (path, value_a, value_b) where two specs differ."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff_specs(a.get(k), b.get(k), f"{path}.{k}" if path else k)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"{len(a)} items", f"{len(b)} items"))
        for i, (x, y) in enumerate(zip(a, b)):
            label = x.get('name', i) if isinstance(x, dict) else i
            out += diff_specs(x, y, f"{path}[{label}]")
    else:
        if a != b and not (_num(a) is not None and _num(b) is not None and math.isclose(_num(a), _num(b))):
            out.append((path, a, b))
    return out
