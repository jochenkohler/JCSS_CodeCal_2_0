# API reference

The package `codecal` has seven modules; the interface (`codecal.gui`)
builds on them and adds nothing numerical. Read them in this order — each
starts with a docstring that explains the concept it implements.

```{eval-rst}
.. automodule:: codecal
   :no-members:
```

## `codecal.distributions`

```{eval-rst}
.. automodule:: codecal.distributions
   :members: Normal, Lognormal, Gumbel, distribution
```

## `codecal.variables`

```{eval-rst}
.. automodule:: codecal.variables
   :members: RandomVariable, rv, characteristic_value_mcs, time_variant_load
```

## `codecal.reliability`

```{eval-rst}
.. automodule:: codecal.reliability
   :members: LimitState, FORMResult, form, sorm, curvatures, SamplingResult, monte_carlo, importance_sampling
```

## `codecal.model`

```{eval-rst}
.. automodule:: codecal.model
   :members: PartialFactors, FailureMode, VariableLoad, CalibrationModel, DesignSituation, evaluate, weighted_mean_beta, beta_table
```

## `codecal.calibrate`

```{eval-rst}
.. automodule:: codecal.calibrate
   :members: objective, bounds_for, CalibrationResult, calibrate
```

## `codecal.spec`

```{eval-rst}
.. automodule:: codecal.spec
   :members: ModelSpecError, validate_spec, build_model, to_json, from_json, spec_signature, diff_specs
```

## `codecal.ahg2021`

```{eval-rst}
.. automodule:: codecal.ahg2021
   :members: ahg_spec, ahg_model
```

## `codecal.plots`

```{eval-rst}
.. automodule:: codecal.plots
   :members: plot_beta_distribution, plot_beta_vs_aQ, plot_sensitivities, plot_form_history, plot_limit_state_2d, plot_calibration_history, plot_gamma_comparison, figure_png
```
