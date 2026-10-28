# The code format: three combination rules

```{admonition} The study in one paragraph
:class: note
During the revision of the Eurocodes, members of the JCSS prepared the
scientific background for a possible reliability-based re-calibration of
the partial factors, within the CEN/TC 250/SC 10 Ad-hoc Group *Reliability
Background of the Eurocodes* {cite}`JRC2024`. The work followed the seven
steps of {doc}`../background/calibration`, with the emphasis on steps 3–7:
the code format (this page), the failure modes and probabilistic models
({doc}`model`), the measure of closeness, the optimal factors and the
verification ({doc}`results`, {doc}`verification`). Sources:
{cite}`KoehlerHingoraniSorensen2021,Kohler2023,KohlerChapter2026`.
```

## Step 3 — the format: equations (8.12), (8.13) and (8.14)

EN 1990:2023 {cite}`EN1990_2023` offers three alternative expressions for
the combination of actions in persistent and transient design situations,
(8.12), (8.13) and (8.14) — the successors of (6.10) and (6.10 a/b) of
EN 1990:2002. National standardisation committees must choose one of them,
and to support that choice the reliability implied by each was needed.

The design situations of the study comprise one permanent load (self-weight
$G_S$ and other permanent load $G_P$) and one variable load $Q_j$. For such
cases the three expressions become *design equations* in which the design
solution $p_{ij}$ — the resistance that the engineer provides for failure
mode $i$ under variable load $j$, e.g. the cross-section area of a tension
rod — is identified as follows.

**(8.12)** — one combination:

$$
p_{ij} = \frac{\gamma_{M,i}}{\theta_{Ri,k}\, r_{i,k}}
\Big\{ (1-a_Q)\big[a_G\,\gamma_{GS}\,g_{S,k} + (1-a_G)\,\gamma_{GP}\,g_{P,k}\big]
      + a_Q\,\theta_{Qj,k}\,\gamma_Q\,q_{j,k} \Big\}
$$

**(8.13)** — the larger of two combinations, the first with the variable
load reduced by $\psi_{0,j}$, the second with the permanent load reduced by
$\xi$:

$$
p_{ij} = \max \begin{cases}
\dfrac{\gamma_{M,i}}{\theta_{Ri,k}\, r_{i,k}}
\Big\{ (1-a_Q)\big[a_G\,\gamma_{GS}\,g_{S,k} + (1-a_G)\,\gamma_{GP}\,g_{P,k}\big]
      + a_Q\,\theta_{Qj,k}\,\psi_{0,j}\,\gamma_Q\,q_{j,k} \Big\} \\[2ex]
\dfrac{\gamma_{M,i}}{\theta_{Ri,k}\, r_{i,k}}
\Big\{ (1-a_Q)\,\xi\big[a_G\,\gamma_{GS}\,g_{S,k} + (1-a_G)\,\gamma_{GP}\,g_{P,k}\big]
      + a_Q\,\theta_{Qj,k}\,\gamma_Q\,q_{j,k} \Big\}
\end{cases}
$$

**(8.14)** — the larger of the permanent load alone and the second
combination of (8.13):

$$
p_{ij} = \max \begin{cases}
\dfrac{\gamma_{M,i}}{\theta_{Ri,k}\, r_{i,k}}
\,(1-a_Q)\big[a_G\,\gamma_{GS}\,g_{S,k} + (1-a_G)\,\gamma_{GP}\,g_{P,k}\big] \\[2ex]
\dfrac{\gamma_{M,i}}{\theta_{Ri,k}\, r_{i,k}}
\Big\{ (1-a_Q)\,\xi\big[a_G\,\gamma_{GS}\,g_{S,k} + (1-a_G)\,\gamma_{GP}\,g_{P,k}\big]
      + a_Q\,\theta_{Qj,k}\,\gamma_Q\,q_{j,k} \Big\}
\end{cases}
$$

Three parts can be recognised in each: the resistance (the fraction in
front), the permanent load (the term in square brackets) and the variable
load (the term with $q_{j,k}$). The second sub-equations of (8.13) and
(8.14) are identical; the first sub-equation of (8.14) neglects the
variable load, and the first of (8.13) reduces it by $\psi_0$. Since the
second combination governs most situations, (8.13) and (8.14) give nearly
the same designs — and both give smaller designs than (8.12).

### Nomenclature

$p_{ij}$
: the design solution for material (failure mode) $i$ and variable load $j$

$\gamma_{M,i}$
: the material partial factor of failure mode $i$ (fixed in the study)

$\theta_{Ri,k}$, $r_{i,k}$
: the characteristic values of the resistance model uncertainty (often 1;
  a bias or hidden safety could be introduced here) and of the material
  strength

$a_Q$
: the share of the variable load in the total load, between 0 (no
  variable load) and 1 (no permanent load)

$a_G$
: the share of the structural self-weight in the permanent load, between 0
  (no self-weight) and 1 (no other permanent load)

$g_{S,k}$, $\gamma_{GS}$
: the characteristic value of the self-weight and its partial factor

$g_{P,k}$, $\gamma_{GP}$
: the characteristic value of the permanent load from non-structural
  elements and its partial factor

$q_{j,k}$, $\theta_{Qj,k}$, $\gamma_Q$
: the characteristic value of variable load $j$, of its model uncertainty,
  and the partial factor on the variable load

$\psi_{0,j}$
: the combination factor of variable load $j$

$\xi$
: the reduction factor on the permanent actions

In CodeCal 2.0, $p_{ij}$ is called the *design parameter* $z$
(`DesignSituation.design_parameter`), and $\gamma_{GS}$, $\gamma_{GP}$ are
`gamma_G` and `gamma_GP` of `PartialFactors`. Following the convention of
the study, calibrating "γ_G" calibrates both permanent-load factors
together unless `gamma_GP` is named separately.

## The limit state function

For all three design equations the same limit state function applies,

$$
g(\mathbf{X}, p_{ij}) = p_{ij}\,\Theta_{R,i}\,R_i
  - \Theta_E \Big[ (1-a_Q)\big(a_G\,G_S + (1-a_G)\,G_P\big) + a_Q\,\Theta_{Q,j}\,Q_j \Big] ,
$$ (eq-lsf)

with the seven random variables

$\Theta_{R,i}$, $R_i$
: the resistance model uncertainty and the strength of material $i$

$\Theta_E$
: the model uncertainty of the load effect (common to all situations)

$G_S$, $G_P$
: the self-weight and the other permanent load

$\Theta_{Q,j}$, $Q_j$
: the model uncertainty and the 50-year maximum of variable load $j$.

A design situation is thus fully described by a failure mode $i$, a
variable load $j$, and the two proportions $(a_G, a_Q)$. The design equation
gives $p_{ij}$, the limit state function with that $p_{ij}$ gives $\beta$
by FORM, and the sweep over all situations gives the picture of the code.

## The benchmark and the calibration

Compliance with the reliability objective is assessed for the three rules
on a representative domain $D$ of design situations: nine failure modes,
three variable loads, and the ranges of $a_Q$ and $a_G$ of the next page.
Two steps:

1. the design solutions $p_{ij}$ are computed with (8.12), (8.13) or (8.14)
   and the present factors ($\gamma_G = 1.35$, $\gamma_Q = 1.5$, $\xi = 0.85$,
   $\psi_0$ = 0.6 / 0.5 / 0.7 for wind / snow / imposed);
2. the reliability index of each solution is computed with the limit state
   function {eq}`eq-lsf`.

The weighted average $\beta_{\text{w-average}} = \sum_k w_k \beta_k$ is the
headline number; the spread of the $\beta_k$ is the story. For the
calibration, an alternative set of action factors
$\boldsymbol{\gamma}_S = (\gamma_G, \gamma_{Q,\text{wind}}, \gamma_{Q,\text{snow}},
\gamma_{Q,\text{imposed}})$ — or just $(\gamma_G, \gamma_Q)$ — is found by
minimising

$$
M(\boldsymbol{\gamma}_S) = \sum_{k \in D} w_k \big(\beta_k(\boldsymbol{\gamma}_S) - \beta_t\big)^2
$$

with $\beta_t = 3.8$, the EN 1990 target for a 50-year reference period
(step 5 and 6 of the procedure). The material factors $\gamma_{M,i}$ are
kept at their Eurocode values throughout.
