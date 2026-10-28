# Learning mode

The learning mode follows the six lessons. Each step has a short
*Background* text (collapsed), the controls on the left, the picture and
the numbers on the right, and a few *Things to try* at the bottom — small
experiments whose outcome teaches something. The model is the AHG model
of {doc}`../eurocode_study/model`; it cannot be edited here (that is what
the lab mode is for).

## Step 1 — Random variables

```{figure} figures/learning_step1.png
:width: 100%
Step 1: a distribution from mean and CoV, its characteristic value, and the u → x transform.
```

Choose a family (lognormal, normal, Gumbel), a mean and a CoV; the
density, the fractile table and the mapping $x = F^{-1}(\Phi(u))$ follow.
The lower panel turns a Gumbel yearly maximum into the maximum over a
reference period and simulates the characteristic value of
$\Theta_Q \cdot Q$ — the quantity that the load tables of the study are
built on.

Things to try: set the CoV of $\Theta_Q$ to zero and watch the simulated
$q_k$ coincide with the 98 % fractile of $Q$ alone; compare the 5 %
fractiles of a lognormal and a normal variable with the same CoV.

## Step 2 — FORM playground

```{figure} figures/learning_step2.png
:width: 100%
Step 2: a two-variable limit state in u-space, the HL-RF iterations, and FORM next to SORM and sampling.
```

Two limit states: $g = R - S$ (nearly linear in u-space when $S$ is
normal — FORM is exact there) and the product $g = c - \Theta Q$, which
bends towards the origin and shows the FORM bias of a few hundredths
that the whole calibration model has. The table lists FORM, SORM and
importance sampling (with its standard error); crude Monte Carlo can be
added. The sensitivities $\alpha$ and the design point are shown with the
iteration history.

## Step 3 — One design situation

```{figure} figures/learning_step3.png
:width: 100%
Step 3: a failure mode, a load, the proportions a_G and a_Q, the design equation and the partial factors — z, β and the α² of the seven variables follow live.
```

The design equation of the Eurocode and the limit state function of the
study meet here. Pick a failure mode, a variable load, $a_G$, $a_Q$, the
combination rule and the partial factors; the *design parameter* $z$,
the reliability index and the share of each variable in the uncertainty
update live, and the right-hand plot shows β against $\gamma_Q$ for
several $a_Q$ — the mechanism that calibration exploits.

## Step 4 — The sweep

```{figure} figures/learning_step4.png
:width: 100%
Step 4: the status quo of the present factors over all design situations.
```

FORM over all design situations (full grid: 810, quick grid: 135) with the
factors you set. The result panel offers five views: β per failure mode
(violins with the weighted means), β per load, β against $a_Q$,
sensitivities $\alpha^2$ per variable, and the mode × load table of mean
β coloured against the target.

## Step 5 — Calibration

```{figure} figures/learning_step5.png
:width: 100%
Step 5: choose the free parameters and the target; the optimiser reports every evaluation.
```

Choose the free parameters (a common $\gamma_G$ and $\gamma_Q$; or
$\gamma_{G,P}$ separately; or one $\gamma_Q$ per load; or ξ for the
two-combination rules), the combination rule, the target and the grid, and
press *Calibrate*. The progress display shows every evaluation of the
objective; the result shows the calibrated factors, the weighted-mean β
before and after, the objective, and the same five views with the start
values as reference. Inconsistent choices (a common and a per-load
$\gamma_Q$ at the same time; ξ with (8.12)) are explained and block the
button rather than producing nonsense.

Things to try: per-load factors (snow gets the largest, imposed the
smallest); a target of 4.3 (consequence class 3); SORM as the method.

## Step 6 — Verification

```{figure} figures/learning_step6.png
:width: 100%
Step 6: FORM against SORM and importance sampling for a sample of the design situations.
```

Every k-th design situation is checked with SORM and importance sampling.
The metrics give the mean and maximum FORM error, the SORM error, and the
share of situations within two standard errors of the sampling result;
the plot shows the error against β; the table groups it by the dominant
variable. This is {doc}`../eurocode_study/verification` for any set of
partial factors.
