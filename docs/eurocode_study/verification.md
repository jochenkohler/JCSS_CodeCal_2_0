# Verification: how good is FORM here?

Step 7 of the procedure. Every reliability index of the study is a
first-order (FORM) value. The limit state function {eq}`eq-lsf` contains
products of lognormal and Gumbel variables, so its failure surface is
curved in standard-normal space — and FORM, which replaces the surface by
its tangent plane at the design point, is then *optimistic* by a small,
systematic amount. {doc}`../../lessons/notebooks/06_verification` runs
SORM (Breitung's curvature correction {cite}`Breitung1984`) and importance
sampling around the design point (20 000 samples, standard error 0.004 in
β) for all 810 situations of the status quo; the table summarises what it
finds.

| | FORM − IS | SORM − IS |
|---|---|---|
| mean over the 810 situations | +0.041 | +0.001 |
| standard deviation | 0.019 | 0.004 |
| range | −0.006 … +0.084 | −0.014 … +0.013 |

The FORM error depends on which variable dominates the design point:
largest (0.05–0.06) when a lognormal variable does — a model uncertainty
or a strength — and smallest (0.015) when the Gumbel load $Q$ does. By
load it is 0.05 for wind and snow and 0.025 for the imposed load. SORM
removes it entirely: the remaining differences are within the sampling
error.

## What it means for the calibration

The weighted-mean β of the status quo is 3.74 by FORM and about 3.70 by
SORM or sampling. Calibrating with SORM instead of FORM moves the
optimum by about +0.01 in both factors (γ_G 1.215 → 1.228, γ_Q 1.750 →
1.761, lesson 6) — a shift of the same order as the resolution of the
numbers reported in the study, and far below the sensitivity to the
modelling assumptions (weights, characteristic values, coefficients of
variation; see the sensitivity study of the lab mode). The AHG study
reported FORM values, and so does this documentation; the tool offers SORM
everywhere at about twice the cost.

The lesson also checks one situation with 200 000 importance samples and
2 million crude Monte-Carlo samples: FORM 3.520, SORM 3.478,
sampling 3.476 ± 0.001 — SORM is right, FORM is 0.04 high, which is the
same picture as over the whole model.

```{admonition} In the tool
:class: tip
The *Verification* step of the learning mode and the *Verification* tab of
the lab mode run this comparison for a sample of the situations of any
model, in a few seconds, with the same functions.
```
