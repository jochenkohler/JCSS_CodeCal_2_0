"""Indicative R vs. E figure: two slightly overlapping densities.

Resistance R is lognormal, load effect E is Gumbel (maxima).

Schematic only - no numbers on either axis, but both curves are true
densities on a common scale, so each encloses unit area. E is given the
larger standard deviation, which shows as the wider, flatter curve.
Vector SVG/PDF with live text, meant to be dropped into a larger
Illustrator graphic.

    python plot_RE_densities.py
    python plot_RE_densities.py --sigma-bars --shade-overlap
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r, lognorm

matplotlib.rcParams["svg.fonttype"] = "none"   # text stays text in the SVG
matplotlib.rcParams["pdf.fonttype"] = 42       # embedded TrueType, editable
matplotlib.rcParams["pdf.compression"] = 0
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

C_E = "#B4472E"   # load effect
C_R = "#1F3B73"   # resistance


def lognormal_from_moments(mean, cov):
    """Frozen scipy lognormal with the given mean and CoV."""
    sigma = np.sqrt(np.log(1.0 + cov ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return lognorm(s=sigma, scale=np.exp(mu))


EULER_GAMMA = 0.5772156649015329


def gumbel_from_moments(mean, cov):
    """Frozen scipy Gumbel (largest values) with the given mean and CoV."""
    beta = mean * cov * np.sqrt(6.0) / np.pi
    return gumbel_r(loc=mean - EULER_GAMMA * beta, scale=beta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mean-e", type=float, default=1.0)
    p.add_argument("--cov-e", type=float, default=0.30)
    p.add_argument("--mean-r", type=float, default=2.6)
    p.add_argument("--cov-r", type=float, default=0.09)
    p.add_argument("--label-e", default="E")
    p.add_argument("--label-r", default="R")
    p.add_argument("--sigma-bars", action="store_true",
                   help="draw a mean +/- 1 sigma bar under each curve")
    p.add_argument("--shade-overlap", action="store_true",
                   help="lightly fill the region where the two densities overlap")
    p.add_argument("--fill", action="store_true", default=True)
    p.add_argument("--no-fill", dest="fill", action="store_false")
    p.add_argument("--out", default="figures/R_E_densities")
    p.add_argument("--size", type=float, nargs=2, default=(4.6, 2.2))
    args = p.parse_args()

    dE = gumbel_from_moments(args.mean_e, args.cov_e)      # load effect: Gumbel
    dR = lognormal_from_moments(args.mean_r, args.cov_r)   # resistance: lognormal

    xmax = max(dE.ppf(0.9995), dR.ppf(0.9995)) * 1.02
    x = np.linspace(1e-9, xmax, 1200)
    yE, yR = dE.pdf(x), dR.pdf(x)   # true densities: each encloses unit area

    fig, ax = plt.subplots(figsize=tuple(args.size))

    if args.shade_overlap:
        ax.fill_between(x, np.minimum(yE, yR), color="#6B6B6B", alpha=0.35, linewidth=0)
    for y, c in ((yE, C_E), (yR, C_R)):
        if args.fill:
            ax.fill_between(x, y, color=c, alpha=0.13, linewidth=0)
        ax.plot(x, y, color=c, linewidth=1.5, solid_capstyle="round")

    if args.sigma_bars:
        ymax_ = max(yE.max(), yR.max())
        for d, c, k in ((dE, C_E, 1), (dR, C_R, 2)):
            m, sd = d.mean(), d.std()
            yb = -0.075 * ymax_ * k
            ax.plot([m - sd, m + sd], [yb, yb], color=c, lw=1.0, clip_on=False,
                    solid_capstyle="butt", marker="|", markersize=4)
            ax.plot([m], [yb], marker="o", ms=2.6, color=c, clip_on=False)

    # curve labels, placed just above each peak
    ymax = max(yE.max(), yR.max())
    for y, c, lab in ((yE, C_E, args.label_e), (yR, C_R, args.label_r)):
        i = int(np.argmax(y))
        ax.text(x[i], y[i] + 0.045 * ymax, lab, color=c, ha="center", va="bottom",
                fontsize=11, fontstyle="italic")

    # bare baseline with an arrow head, no numbers anywhere
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.20 * ymax if args.sigma_bars else 0, ymax * 1.28)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.spines["bottom"].set_position(("data", 0.0))   # baseline at f(x) = 0
    ax.spines["bottom"].set_color("#333333")
    ax.plot(1, 0, marker=">", transform=ax.get_yaxis_transform(), clip_on=False,
            markersize=4, color="#333333")
    ax.set_xlabel(f"{args.label_e},  {args.label_r}", fontsize=10, fontstyle="italic",
                  loc="right", labelpad=2)

    fig.tight_layout()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(f"{out}.{ext}", transparent=True)
        print(f"wrote {out}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
