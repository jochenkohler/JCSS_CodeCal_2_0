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

The same story can be followed by clicking: `streamlit run app.py` opens a
graphical interface with a **learning mode** (the six lessons as interactive
steps) and a **lab mode** (edit the model, calibrate, compare, test the
assumptions). See [The graphical interface](#the-graphical-interface).

**Documentation:** <https://jochenkohler.github.io/JCSS_CodeCal_2_0/> — background,
the Eurocode study reproduced, the manual, the lessons, the API.
**The interface in the browser, nothing to install:**
<https://jochenkohler.github.io/JCSS_CodeCal_2_0/app/>.

## Setup

Python ≥ 3.10 with numpy, scipy, pandas and matplotlib.

```bash
cd JCSS_CodeCal_2_0
pip install -e ".[dev]"        # codecal + streamlit, pytest, jupytext; or just: pip install -r requirements.txt
pytest                         # 50 tests, ~2 min — includes the reproduction of the AHG numbers and the GUI
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
python run_calibration.py --model my_model.json --free gamma_G gamma_Q   # a model saved by the lab mode
```

## The graphical interface

```bash
streamlit run app.py           # opens http://localhost:8501 in the browser
```

**Learning mode** — the six lessons as interactive steps, each with a short
background text, controls on the left, the picture and the numbers on the
right, and a few things to try:

1. *Random variables*: distribution, mean, CoV → density, the u → x transform, the fractile table; a Gumbel yearly maximum → T-year maximum and the simulated characteristic load.
2. *FORM playground*: a two-variable limit state (R − S, or the product c − Θ·Q) in u-space, the HL-RF iterations, sensitivities, SORM and importance sampling next to FORM.
3. *One design situation*: pick a failure mode, a load, a_G, a_Q, the design equation and the partial factors — z, β and the α² of the seven variables follow live; β against γ_Q for several a_Q.
4. *The sweep*: the status quo of the present factors over all design situations — β per mode, per load, against a_Q, sensitivities, and the mode × load table.
5. *Calibration*: choose the free parameters and the target, watch the optimiser work, compare calibrated and start values.
6. *Verification*: FORM against SORM and importance sampling over a sample of the model.

**Lab mode** — the research tool, six tabs:

* *Model*: every number of the probabilistic model in two editable tables (failure modes, variable loads) plus the global settings; add or remove rows, pin a characteristic value or give a fractile, untick a row to exclude it. The model is validated as you type — problems are listed in words, and the run buttons stay hidden until it is consistent. Save it as JSON, load it back later, or reset to the AHG 2021 model (the differences from AHG are listed).
* *Status quo*: β for every design situation for any set of partial factors (common or per-load γ_Q, FORM or SORM, eq. 8.12 / 8.13 / 8.14), with the same views as the learning mode and a CSV download.
* *Calibration*: free parameters, start values, bounds, target (the model's β_t or the status-quo mean), optimiser settings; a progress display, the optimiser history, a warning when a factor sits on a bound.
* *Compare*: every status-quo or calibration result can be saved as a named snapshot; overlay any two, and see the calibrated factors of all calibrations side by side.
* *Sensitivity*: vary one parameter of the model (a CoV, a bias, a weight, the basic period, the target …) over a range and watch the status-quo reliability — or the calibrated γ_G, γ_Q — respond.
* *Verification*: as in the learning mode, with the edited model.

Models (`*.json`) are plain text: `codecal.from_json` / `codecal.build_model`
read them in a script, so a model set up in the lab can be calibrated with
`run_calibration.py` and the lessons' functions. Results are cached per model
and factor set, so repeated runs are instant.

Errors never leave a blank page: an inconsistent model is reported as a
list of problems, a failed computation as one readable message with the
details in an expander.

## Documentation and the web version

The documentation is a Jupyter Book (`_config.yml`, `_toc.yml`, `docs/`) in the
same style as the JCSS Probabilistic Model Code. It has three parts — the
background (codes as decision rules, calibration as optimisation, the reliability
concepts), the Eurocode study (the code format, the probabilistic model generated
from `ahg_spec()`, the results of the AHG study reproduced with this code and
compared number by number, the verification), and the manual of the tool — plus
the six lessons as executed notebooks and the API reference.

```bash
pip install "jupyter-book<2" sphinxcontrib-bibtex
make book          # → _build/html/index.html
make notebooks     # re-execute the lessons and the documentation notebooks (≈ 15 min)
make web           # → web/codecal_app.zip, the interface for the browser (stlite)
make site          # book + web as GitHub Pages serves them
```

Pushing to `main` runs `.github/workflows/book.yml`, which builds the book and the
web bundle and publishes them to the `gh-pages` branch: the book at the root of the
Pages site, the interface at `/app/` (Streamlit running on Pyodide in the browser —
see `web/README.md`; the installed version is several times faster, and
[Streamlit Community Cloud](https://share.streamlit.io) can host `app.py` at full
speed).

## The package

```
codecal/
├── distributions.py   Normal, Lognormal, Gumbel from (mean, CoV); ppf, cdf, x(u), dx/du; Gumbel period shift
├── variables.py       RandomVariable with characteristic value; time-variant loads and their q_k by simulation
├── reliability.py     LimitState, FORM (HL-RF), SORM (Breitung), crude Monte Carlo, importance sampling
├── model.py           FailureMode, VariableLoad, PartialFactors, DesignSituation (design equations, limit state),
│                      CalibrationModel, evaluate() = the sweep
├── calibrate.py       the objective Σ w (β − β_t)² and its SLSQP minimisation
├── spec.py            the model as a plain dictionary: validation, build, JSON save / load, diff
├── ahg2021.py         the AHG model as data — every number of the study in one readable file
├── plots.py           the figures
└── gui/               the Streamlit app: common.py (cache, guards, panels), learning.py, lab.py
app.py                 streamlit run app.py
docs/                  the Jupyter Book (with lessons/notebooks/ and the API reference)
web/                   the interface as a static web page (stlite)
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

## Licence

Code: MIT. Documentation and lessons: CC BY 4.0. See `LICENSE`. The model is
the input of the CEN/TC 250/SC 10 Ad-hoc Group *Reliability Background of the
Eurocodes* (2021), published in the JRC Technical Report of 2024 (CC BY 4.0);
the calibration study is the work of Jochen Köhler, Ramon Hingorani and John
Dalsgaard Sørensen. If you use CodeCal 2.0 in a publication, please cite the
JRC report and the repository.
