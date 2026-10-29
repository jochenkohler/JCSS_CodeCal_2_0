# Reliability-based code calibration

## What calibration is

In the structural reliability literature, code calibration is "the process
of assigning values to the parameters in a design code … with a view to
achieve a desired level of reliability in error-free structures"
{cite}`MadsenKrenkLind1986`. The parameters are the *reliability elements*
of the code: partial factors, load reduction and combination factors,
modification factors — or, in a probabilistic code, explicit target failure
probabilities.

Three strategies are distinguished {cite}`DitlevsenMadsen1996,MadsenKrenkLind1986`:

Judgement
: adjusting the reliability elements from experience and observation, by
  trial and error. Not suitable for new materials, technologies or structural
  concepts, for which there is no experience.

Fitting
: calibrating a new code such that it gives essentially the same reliability,
  or the same designs, as an existing code considered acceptable. Common when
  an existing code is rewritten in a new format
  {cite}`SorensenHansenNielsen2001,HendersonBlockley1982`.

Optimisation
: establishing acceptance criteria, determining the optimal design from a
  utility function (expected benefits and costs), and calibrating the
  reliability elements to it
  {cite}`FaberSorensen2003,KohlerSorensenBaravalle2019,Kohler2023,Stewart2016`.

Calibration is carried out by committees, and many of the decisions rely on
discussion and consensus — subjective factors, represented experience and
judgement. Rational methods fall short where opinions differ. A significant
part of the work can nevertheless be based on objective reasoning, and a
practical procedure in seven steps has been proposed
{cite}`Ellingwood1980,FaberSorensen2003,MadsenKrenkLind1986,ThoftChristensenBaker1982`
and standardised in ISO 2394:2015 {cite}`ISO2394`:

1. Definition of the scope of the code.
2. Definition of the code objective(s).
3. Definition of the code format.
4. Identification of typical failure modes and probabilistic models.
5. Definition of a measure of closeness to the reliability criterion.
6. Determination of the optimal partial factors for the chosen format.
7. Verification.

For the revision of an existing standard, steps 3 to 7 are the relevant
ones. The core is step 6, the selection of the partial factors by solving an
optimisation problem — and the Eurocode study in the next part follows
exactly these steps.

## Calibration as an optimisation problem

A semi-probabilistic code consists of a set of reliability elements
$\mathbf{r}$ — partial factors, combination factors, characteristic values.
Traditionally these are calibrated to *reliability* rather than to risk: the
simplified objective is a consistent reliability level across the design
situations covered by the code. Following Siu, Parimi and Lind
{cite}`Siu1975`, the optimal set is the one that minimises the squared
deviation between the reliability $\beta_i(\mathbf{r})$ achieved in design
situation $i$ and a target $\beta_t$,

$$
\mathbf{r}^{*} = \arg\min_{\mathbf{r}} \sum_i w_i \big(\beta_i(\mathbf{r}) - \beta_t\big)^2 ,
$$ (eq-penalty)

where the weights $w_i$ express how frequent, or how important, the design
situations are. The sum is called the *penalty function* or the *objective*;
constraints may be added for minimum reliability levels related to human
safety. Several forms of the penalty function have been proposed (see
{cite}`Baravalle2017` for an overview), and it turns out that its exact
mathematical form has limited influence on the result. The more
consequential modelling decision is the choice of $\beta_t$.

### The target reliability

Three interpretations are met in practice:

*A prescriptive requirement.* A fixed reliability is set by the authorities
and used directly. This is common, but it can be misleading: reliability and
failure probability are not inherent properties of a structure; they are
attributes of the analysis model used to represent uncertainty and
performance.

*A requirement derived from optimisation.* The target is derived from a
risk-informed analysis of uncertainties and consequences — in practice
simplified by treating the calibration scenarios as typical cases with
moderate consequences and an average relative cost of safety measures, as
ISO 2394 does when it recommends $\beta \approx 4.2$ for a one-year reference
period for such cases. The difficulty is that semi-probabilistic design
expressions have evolved over long periods and carry implicit conservatism
and model bias ("hidden safety") that is hard to quantify and differs
between expressions; an absolute calibration to a single target may then not
produce the intended outcome.

*A relative calibration to an accepted code.* If an existing code is
regarded as acceptable and economical, the objective becomes to reduce the
*variability* of the reliability across the design situations, taking the
average reliability of the existing code as the target
{cite}`BaravalleKohler2017,Baravalle2017`. This is less sensitive to the
modelling assumptions and to model bias, as far as these affect the
reliability elements in a comparable way; the absolute safety level,
however, cannot be assessed this way.

All three have limitations, since the design situations regulated by a code
are non-homogeneous in regard to model uncertainties and hidden safety. The
Eurocode study uses the prescribed target of EN 1990 ($\beta_t = 3.8$ for a
50-year reference period) and reports the results conditional on the model;
the lab mode of the tool offers the status-quo average as an alternative
target.

## The design value method

The *design value format* is a simplified way of deriving partial factors
without any reliability analysis; it is the method behind the recommended
values of EN 1990 and ISO 2394. The design value $x_d$ of a variable $X$
with distribution function $F_X$ is

$$
x_d = F_X^{-1}\big(\Phi(\alpha_X \beta_t)\big),
$$ (eq-design-value)

where $\alpha_X$ (with $|\alpha_X| \le 1$) is the FORM sensitivity factor of
$X$ — the share of $X$ in the uncertainty at the design point — and
$\Phi$ the standard normal distribution function. The codes fix
$\alpha_R = -0.8$ for the resistance and $\alpha_E = 0.7$ for the dominating
load variable, valid as long as the ratio of the standard deviations of load
and resistance lies between 0.16 and 7.6. With the characteristic values
defined as specified fractiles — 5 % for resistances, the mean for permanent
actions, the 98 % fractile of the annual maximum for variable actions — the
partial factor is the ratio $x_d / x_k$.

The method is attractive because the factors for loads and for resistances
can be derived separately. Its weakness is that the real sensitivity factors
depend strongly on the coefficients of variation: with the fixed
$\alpha$-values the method yields conservative factors in many cases —
sometimes very conservative, with excessive material use — and too low
reliability in others, and the variability of the achieved reliability
across load ratios and materials is much larger than for factors calibrated
by optimisation {cite}`Gayton2004`. Model uncertainties are also difficult
to include consistently. Lesson 3 and the learning mode of the tool let you
compare the two approaches on the Eurocode model.
