# JCSS CodeCal 2.0

**Reliability-based calibration of partial factors — the method, the Eurocode
study, and a tool to repeat it.**

Semi-probabilistic design codes such as the Eurocodes ask the engineer to
verify that a factored resistance exceeds a factored load effect. The partial
factors in that inequality are not arbitrary: they are, in principle, the
outcome of a *calibration* — an optimisation in which the factors are chosen
such that the structures designed with them reach a prescribed reliability as
uniformly as possible. CodeCal 2.0 is a small, readable implementation of this
calibration, built around the probabilistic model that the CEN/TC 250/SC 10
Ad-hoc Group *Reliability Background of the Eurocodes* agreed on for the
second generation of the Eurocodes {cite}`JRC2024`.

The documentation has three parts:

**Background** — a short account of why codes are calibrated, how calibration
is posed as an optimisation problem, and the few reliability concepts needed
to follow it. It condenses {cite}`KohlerSorensenEllingwood2025` and
{cite}`KohlerChapter2026`.

**The Eurocode study** — the code format (the combination rules (8.12),
(8.13) and (8.14) of EN 1990:2023), the probabilistic model, and the results
of the benchmark and calibration carried out for the Ad-hoc Group
{cite}`KoehlerHingoraniSorensen2021,Kohler2023` — reproduced here with
CodeCal 2.0, so every number on these pages can be regenerated.

**The tool** — a manual for the graphical interface (a *learning mode* that
walks through the method in six steps and a *lab mode* for one's own models
and calibrations), the model files, and the Python package `codecal` for
scripting. The six lessons are included as executed notebooks.

```{admonition} Where to start
:class: tip
* Curious about the method: read the three background chapters (half an hour), then try the learning mode.
* Looking for the numbers of the Eurocode study: go to {doc}`eurocode_study/results`.
* Want to calibrate your own model: {doc}`manual/getting_started`, then {doc}`manual/lab_mode`.
```

The code is on GitHub at
[jochenkohler/JCSS_CodeCal_2_0](https://github.com/jochenkohler/JCSS_CodeCal_2_0);
the graphical interface runs in the browser at
[jochenkohler.github.io/JCSS_CodeCal_2_0/app](https://jochenkohler.github.io/JCSS_CodeCal_2_0/app/)
(no installation, everything is computed on your machine; a calibration takes
a few minutes there — the installed version is several times faster).

```{admonition} Provenance
:class: note
CodeCal 2.0 is a rewrite of the calibration code used for the Ad-hoc Group's
study (Hingorani, Köhler, Sørensen, 2021–2023), restructured for reading and
teaching. It reproduces the results of that code to the numerical precision
of the optimiser; the comparison with the published numbers is on the results
page. Jochen Köhler, NTNU, 2026. Code under the MIT licence, this text
under CC BY 4.0.
```
