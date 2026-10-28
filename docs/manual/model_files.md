# Model files (JSON)

The lab mode saves and loads the probabilistic model as a JSON file; the
same file is read by `codecal.from_json` in a script and by the
command line (`run_calibration.py --model my_model.json`). It is plain
text, so it can be versioned, diffed and edited by hand.

## Structure

```text
{
 "name": "AHG 2021 — Reliability Background of the Eurocodes",
 "T_ref": 50, "n_aQ": 10, "n_aG": 3, "beta_target": 3.8,
 "seed": 12345, "n_mcs": 10000000,
 "theta_E": {"name": "Load-effect model uncertainty", "family": "lognormal",
             "mean": 1.0, "cov": 0.1, "char": 1.0, "note": "by definition"},
 "P": {"name": "Permanent load (other than self-weight)", "family": "normal",
       "mean": 1.0, "cov": 0.1, "fractile": 0.5},
 "modes": [
  {"name": "Steel bending",
   "theta_R": {"name": "Θ_R steel bending", "family": "lognormal", "mean": 1.15, "cov": 0.05, "char": 1.0},
   "R": {"name": "Steel yield strength", "family": "lognormal", "mean": 1.0, "cov": 0.05, "char": 0.83},
   "gamma_M": 1.0,
   "G": {"name": "Self-weight steel", "family": "normal", "mean": 1.0, "cov": 0.025, "char": 1.0},
   "a_Q": [0.3, 0.8], "a_G": [0.6, 1.0], "weight": 0.17, "include": true},
  ...
 ],
 "loads": [
  {"name": "wind",
   "theta_Q": {"name": "Wind model uncertainty", "family": "lognormal", "mean": 0.97, "cov": 0.26, "char": 1.0},
   "mean": 1.0, "cov": 0.14, "T_input": 50, "T_basic": 1,
   "fractile": 0.98, "char": null, "psi_0": 0.6, "weight": 0.4, "include": true},
  ...
 ]
}
```

A random variable (`theta_E`, `P`, `theta_R`, `R`, `G`, `theta_Q`) is a
dictionary with `family` (`normal`, `lognormal` or `gumbel`), `mean`,
`cov`, and **either** `char` (a pinned characteristic value) **or**
`fractile` (the characteristic value is that fractile); `name` and `note`
are optional and appear in the tables of the interface.

A failure mode has a `name`, the three variables `theta_R`, `R`, `G`, the
material factor `gamma_M`, the ranges `a_Q` and `a_G` as `[min, max]`, a
`weight`, and `include`.

A variable load has a `name`, the model uncertainty `theta_Q`, the
moments of the maximum over `T_input` years (`mean`, `cov`), the basic
period `T_basic`, the `fractile` that defines $q_k$ over the basic period,
an optional pinned `char` (`null` to simulate), `psi_0`, `weight` and
`include`.

Global: `T_ref` (reference period), `n_aQ` and `n_aG` (grid), `beta_target`,
`seed` and `n_mcs` (the Monte-Carlo simulation of the characteristic
loads; the seed makes it reproducible).

## Validation

`codecal.validate_spec(spec)` returns a list of problems in words — an
empty list means `build_model` will succeed. The lab mode runs it on every
change; `from_json` runs it on load and raises `ModelSpecError` with the
list. Checks include the ranges of every number, that each variable has
exactly one of `char`/`fractile`, unique names, at least one included
mode and one included load with positive weight, and consistent `a_Q`/`a_G`
ranges.

## Working with model files

```python
import codecal as cc

spec = cc.ahg_spec()                      # the default model as a dictionary
spec['loads'][1]['theta_Q']['cov'] = 0.30 # more uncertain snow model
spec['modes'][0]['include'] = False       # leave steel out
print(cc.validate_spec(spec))             # [] if consistent
print(cc.diff_specs(cc.ahg_spec(), spec)) # what changed: [('loads[snow].theta_Q.cov', 0.26, 0.3), ...]

with open('snow_study.json', 'w') as f:
    f.write(cc.to_json(spec))
spec2 = cc.from_json(open('snow_study.json').read())
model = cc.build_model(spec2)             # a CalibrationModel, ready for evaluate() and calibrate()
```

`spec_signature(spec)` gives a canonical JSON string that the interface
uses as a cache key: two specs with the same signature build the same
model.
