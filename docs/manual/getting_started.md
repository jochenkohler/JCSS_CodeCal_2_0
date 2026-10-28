# Getting started

CodeCal 2.0 is a Python package (`codecal`), a graphical interface
(`app.py`, Streamlit), six lessons, and a command-line script. There are
three ways to use it.

## In the browser, without installing anything

The interface runs as a web page at
[jochenkohler.github.io/JCSS_CodeCal_2_0/app](https://jochenkohler.github.io/JCSS_CodeCal_2_0/app/).
Python and the numerical libraries are downloaded into the browser on
first load (about 40 MB, a minute on a normal connection) and everything
is computed on your machine; nothing is sent anywhere. It is the same
code as the installed version, only slower — a sweep over the 810 design
situations takes a few seconds, a calibration a few minutes. Fine for the
learning mode and for trying things; for a research session, install it.

## Installed

Python 3.10 or newer. Then:

```bash
git clone https://github.com/jochenkohler/JCSS_CodeCal_2_0.git
cd JCSS_CodeCal_2_0
pip install -e ".[dev]"          # codecal with numpy, scipy, pandas, matplotlib, streamlit, pytest, jupytext
pytest                           # 50 tests, about two minutes; includes the reproduction of the AHG numbers
streamlit run app.py             # the interface opens in the browser at http://localhost:8501
```

`pip install -r requirements.txt` instead of the editable install also
works if you only want to run things.

The lessons are plain Python files with cell markers (`# %%`) in
`lessons/`: open them in Spyder or VS Code and run cell by cell, or read
the executed notebooks in `lessons/notebooks/` (also in this
documentation, part *The lessons*).

## The command line

A complete calibration without reading anything:

```bash
python run_calibration.py                                     # eq. 8.12, γ_G and γ_Q
python run_calibration.py --equation 8.13 --free xi gamma_G gamma_Q
python run_calibration.py --free gamma_G gamma_Q:wind gamma_Q:snow gamma_Q:imposed --method sorm
python run_calibration.py --help
```

## Which mode?

| | Learning mode | Lab mode |
|---|---|---|
| for | first contact with the method; teaching | your own models, calibrations and sensitivity studies |
| model | the AHG model, fixed | editable, save/load as JSON |
| structure | six steps, following the lessons | six tabs: Model, Status quo, Calibration, Compare, Sensitivity, Verification |
| results | per step | snapshots that can be compared and downloaded |

Switch between them in the sidebar. Both use the same functions as the
lessons and the command line, so any number you see in the interface can
be reproduced in a script ({doc}`scripting`).

## Running time

On a laptop, one evaluation of the 810 design situations takes about a
second with FORM (twice that with SORM); a calibration needs 15–75
evaluations. The learning mode offers a *quick grid* (one value of $a_G$,
five of $a_Q$: 135 situations) that gives results in seconds and comes
close to the full grid. The characteristic loads are simulated when a
model is built (10 million draws in the tests and the command line, 1
million in the interface, where it makes a difference in the third
decimal); models and sweeps are cached, so repeating a run costs nothing.
