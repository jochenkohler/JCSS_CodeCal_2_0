# The reliability concepts needed

```{admonition} Reading time
:class: note
Ten minutes. This is the minimum needed to follow the Eurocode study; the
lessons ({doc}`../../lessons/notebooks/01_random_variables` and
{doc}`../../lessons/notebooks/02_form`) treat the same material with code
and exercises, and the compendium {cite}`Kohler2026Compendium` or
{cite}`Melchers2018` give the full account.
```

## Limit state, failure probability, reliability index

A failure mode is described by a *limit state function* $g(\mathbf{x})$ of
the basic variables $\mathbf{x}$ — strengths, loads, model uncertainties —
with $g \le 0$ meaning failure. The probability of failure is

$$
p_f = P\big(g(\mathbf{X}) \le 0\big) .
$$

For a linear $g$ of normally distributed variables, the safety margin
$M = g(\mathbf{X})$ is normal and $p_f = \Phi(-\beta)$ with
$\beta = \mu_M / \sigma_M$ — the *reliability index*. In general $\beta$ is
simply defined through $p_f$,

$$
\beta = -\Phi^{-1}(p_f),
$$

and is the quantity the codes prescribe: EN 1990 asks for $\beta = 3.8$ for
a 50-year reference period in the ordinary consequence class, which
corresponds to $p_f \approx 7 \cdot 10^{-5}$.

## FORM: the design point and the sensitivity factors

Each basic variable is mapped to a standard normal variable,
$x_i = F_{X_i}^{-1}\big(\Phi(u_i)\big)$, so that the problem lives in a
space where the probability density is rotationally symmetric and decays
with the distance from the origin. The *first-order reliability method*
(FORM) {cite}`HasoferLind1974,RackwitzFiessler1978` finds the point of the
failure surface $g(\mathbf{u}) = 0$ closest to the origin — the *design
point* $\mathbf{u}^*$ — and takes its distance as the reliability index:

$$
\beta = \|\mathbf{u}^*\|, \qquad p_f \approx \Phi(-\beta).
$$

The approximation replaces the failure surface by its tangent plane at the
design point. It is exact for a linear surface and remains very good for
the smooth, mildly curved surfaces of code calibration; the error can be
checked with a second-order correction (SORM, {cite}`Breitung1984`) or with
sampling around the design point, which is what the verification pages do.

The unit vector $\boldsymbol{\alpha} = \mathbf{u}^* / \beta$ carries the
*sensitivity factors*: $\alpha_i^2$ is the share of variable $i$ in the
total uncertainty at the design point, and $\sum \alpha_i^2 = 1$. Load
variables have positive $\alpha$ (a larger load brings failure closer),
resistance variables negative. The *design values* are the coordinates of
the design point mapped back,
$x_{i,d} = F_{X_i}^{-1}\big(\Phi(\alpha_i \beta)\big)$ — which is the
relation the design value method of the previous chapter generalises with
fixed $\alpha$.

## Characteristic values and partial factors

The code works with one *characteristic value* per variable — a specified
fractile: 5 % for material strengths, the mean (50 %) for self-weight,
the 98 % fractile of the annual maximum for climatic actions — and a partial
factor that moves it towards the design value, $x_d = x_k \gamma$ for a
load, $x_d = x_k / \gamma$ for a resistance. A design made with the code
provides a certain resistance; FORM on the limit state function then tells
which $\beta$ that design actually reaches. Calibration closes the loop by
choosing the $\gamma$ such that the $\beta$'s come out as intended.

## Normalised variables

In the Eurocode model every variable has mean 1 — the absolute magnitude of
a strength or a load does not matter for the comparison, only its
coefficient of variation (CoV), its distribution family and its
characteristic value as a ratio to the mean. A resistance is then described
by three numbers (family, CoV, $x_k$), a load likewise, and the whole
probabilistic model of the study fits in two tables.

## Time-variant loads

Wind and snow are represented by the distribution of their annual maximum
(a Gumbel distribution), the imposed load by that of its five-year maximum.
The reliability is required for a 50-year reference period, so the
distribution of the maximum over 50 years is needed: for a Gumbel variable
with scale $\sigma$ the standard deviation is unchanged and the mean shifts
by $(\sqrt{6}/\pi)\,\sigma \ln(T_2/T_1)$. The characteristic value of a
variable load is a fractile of the annual maximum of the *product* of the
load and its model uncertainty $\Theta_Q$, which is why it is obtained by
simulation in the model ({doc}`../eurocode_study/model`).

## The weighted average and the objective

The calibration domain is a set of *design situations* — a failure mode, a
dominating variable load, and the proportions $a_G$ (share of self-weight
in the permanent load) and $a_Q$ (share of the variable load in the total)
— each with a weight $w_k$. Two numbers summarise a set of partial factors:
the weighted average reliability

$$
\beta_{\text{w-average}}(\boldsymbol{\gamma}) = \sum_k w_k\, \beta_k(\boldsymbol{\gamma}),
$$

and the objective

$$
M(\boldsymbol{\gamma}) = \sum_k w_k \big(\beta_k(\boldsymbol{\gamma}) - \beta_t\big)^2 ,
$$

which the calibration minimises. The first says whether the code is on
target on average; the second — or the standard deviation of the $\beta_k$
— says how *uniform* the reliability is, which is the more robust
indicator of the quality of a code format.
