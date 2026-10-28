# JCSS CodeCal 2.0 — reliability-based code calibration, written to be read

A small, self-contained Python implementation of the calibration of partial
safety factors (γ_G, γ_Q, ξ) of EN 1990, built on the probabilistic model
of the AHG *Reliability Background of the Eurocodes* (2021). It reproduces
the numbers of the original AHG software to six decimals — and is meant to
be understood in an afternoon.

The code is organised as a short course. Six lessons, each a Python file
you run cell by cell, take you from "what is a characteristic value" to
"which γ_Q does the Eurocode need, and how sure are we":

| Lesson | Topic | Runtime |
|---|---|---|
| `lessons/01_random_variables.py` | distributions from mean and CoV, fractiles, the u-space transform, 50-year maxima, characteristic values by simulation | seconds |
| `lessons/02_form.py` | the reliability index, FORM step by step on a two-variable problem, sensitivities and design values, SORM, importance sampling | ~20 s |
| `lessons/03_design_situation.py` | the design equation, the design parameter z, one design situation of the AHG model taken apart, the three equations 8.12 / 8.13 / 8.14 | seconds |
| `lessons/04_the_sweep.py` | the 810 design situations, the status quo of the present Eurocode, where reliability is high and low, what drives it | seconds |
| `lessons/05_calibration.py` | the objective function as a landscape, calibrating γ_G and γ_Q, more degrees of freedom, the two-combination rule | ~3 min |
| `lessons/06_verification.py` | FORM against SORM and importance sampling over the whole model; the method decision | ~2 min |

Every lesson ends with exercises. Figures are written to `lessons/figures/`.

## Setup

Python ≥ 3.10 with numpy, scipy, pandas and matplotlib.

```bash
cd JCSS_CodeCal_2_0
pip install -e ".[dev]"        # installs codecal + pytest/jupytext; or just: pip install numpy scipy pandas matplotlib
pytest                         # 21 tests, ~30 s — includes the reproduction of the AHG numbers
```

## Three ways to run a lesson

* **Spyder** (recommended for a first read): open the file, run it cell by cell with *Ctrl+Enter* — the `# %%` lines are cell borders and the `# %% [markdown]` cells are the text.
* **Jupyter**: `lessons/notebooks/` holds the same lessons as executed notebooks (outputs and figures included, so they can be read without running anything — on GitHub, in VS Code or in JupyterLab). To regenerate them after editing a lesson: `cd lessons && bash build_notebooks.sh`.
* **Plain Python**: `python lessons/04_the_sweep.py` — prints the results and saves the figures.

A complete calibration without reading anything:

```bash
python run_calibration.py                                     # eq. 8.12, γ_G and γ_Q
python run_calibration.py --equation 8.13 --free xi gamma_G gamma_Q
python run_calibration.py --free gamma_G gamma_Q:wind gamma_Q:snow gamma_Q:imposed --method sorm
```

## The package

```
codecal/
├── distributions.py   Normal, Lognormal, Gumbel from (mean, CoV); ppf, cdf, x(u), dx/du; Gumbel period shift
├── variables.py       RandomVariable with characteristic value; time-variant loads and their q_k by simulation
├── reliability.py     LimitState, FORM (HL-RF), SORM (Breitung), crude Monte Carlo, importance sampling
├── model.py           FailureMode, VariableLoad, PartialFactors, DesignSituation (design equations, limit state),
│                      CalibrationModel, evaluate() = the sweep
├── calibrate.py       the objective Σ w (β − β_t)² and its SLSQP minimisation
├── ahg2021.py         the AHG model as data — every number of the study in one readable file
└── plots.py           the figures
```

Each module starts with a docstring that explains the concept it implements;
the modules are short enough to read top to bottom.

## The theory in one page

**Probabilistic model.** Every quantity is a random variable with mean 1
(normalised) and a coefficient of variation. Resistances and model
uncertainties are lognormal, permanent loads normal, the maxima of
time-variant loads over the reference period Gumbel. The maximum over
T₂ years of a yearly Gumbel maximum is Gumbel with the same σ and the mean
shifted by (√6/π) σ ln T₂.

**Characteristic values** are fractiles: 5 % for strengths, the mean for
permanent loads, the 98 % fractile of the yearly maximum for climatic loads
(taken on the product of the load and its model uncertainty).

**Design equation (EN 1990 eq. 8.12).** With load fractions a_Q (variable
share) and a_G (self-weight share of the permanent part),

    z θ_R,k r_k / γ_M = (1 − a_Q) [a_G γ_G g_k + (1 − a_G) γ_GP p_k] + a_Q γ_Q q_k .

z is the design parameter — the mean resistance the engineer provides per
unit of mean load. Equations 8.13 and 8.14 take the less favourable of two
expressions with ψ₀ on the variable load and ξ on the permanent load.

**Limit state.**  g = z Θ_R R − Θ_E [(1 − a_Q)(a_G G + (1 − a_G) P) + a_Q Θ_Q Q].

**FORM.** In the space of independent standard normals u (x_i = F_i⁻¹(Φ(u_i))),
the design point u* is the point of g(u) = 0 closest to the origin;
β = ‖u*‖, P_f ≈ Φ(−β), α = u*/β are the sensitivities. Found by the
HL-RF iteration u_{k+1} = [α_k·u_k + g(u_k)/‖∇g(u_k)‖] α_k.

**SORM** (Breitung) corrects for the curvatures κ_i of the surface at u*:
P_f ≈ Φ(−β) Π (1 + β κ_i)^{−1/2}. For this limit state FORM is optimistic by
≈ 0.04 in β.

**Calibration.** Over the design situations k (mode × load × a_G × a_Q) with
weights w_k, minimise M(γ) = Σ w_k (β_k(γ) − β_t)² with β_t = 3.8 (50 years,
consequence class 2).

## Results to expect (AHG model, eq. 8.12, FORM)

| | γ_G | γ_Q | weighted-mean β | M |
|---|---|---|---|---|
| Eurocode (start values) | 1.35 | 1.50 | 3.741 | 0.189 |
| calibrated, γ_G and γ_Q | 1.215 | 1.750 | 3.809 | 0.1366 |

The status-quo table of mean β per failure mode and load (lesson 4)
matches the AHG report; with SORM the status-quo mean is 3.705 and the
calibrated factors come out about 0.01–0.02 higher.

## Provenance

The model and the numbers come from the AHG input of November 2021 (T.
Vrouwenvelder et al.) as implemented by R. Hingorani and J. Köhler
(github.com/RamonHingorani/Eurocode-Calibration) and revised in the
Eurocode-Calibration-Workbench (October 2026). `tests/reference_workbench.json`
holds the workbench's results; `pytest` checks that this implementation
reproduces them (β to 2·10⁻⁶, the calibrated factors to 10⁻³).

Open questions carried over from the AHG input: the pinned characteristic
values of steel (0.83) and aluminium (0.85) strength; FORM versus SORM for
the published factors (lesson 6).
