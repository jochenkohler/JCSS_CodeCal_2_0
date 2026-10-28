"""Vector (SVG/PDF) plot of a lognormal density, ready for Illustrator.

Text is exported as real text (not outlines) and no element is rasterised,
so every curve, tick and label stays editable after import.

    python plot_lognormal_pdf.py                 # mean 1.0, CoV 0.30
    python plot_lognormal_pdf.py --mean 1 --cov 0.15 --out figures/R
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

# --- keep the export editable ------------------------------------------------
matplotlib.rcParams["svg.fonttype"] = "none"   # text stays text in the SVG
matplotlib.rcParams["pdf.fonttype"] = 42       # embedded TrueType, editable
matplotlib.rcParams["pdf.compression"] = 0
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]


def lognormal_from_moments(mean, cov):
    """Return a frozen scipy lognormal with the given mean and CoV."""
    sigma = np.sqrt(np.log(1.0 + cov ** 2))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return lognorm(s=sigma, scale=np.exp(mu))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mean", type=float, default=1.0, help="mean of X")
    p.add_argument("--cov", type=float, default=0.30, help="coefficient of variation of X")
    p.add_argument("--xmax", type=float, default=None, help="upper x limit (default: 99.9%% quantile)")
    p.add_argument("--fill", action="store_true", help="fill the area under the density")
    p.add_argument("--label", default="x", help="x-axis label")
    p.add_argument("--out", default="figures/lognormal_pdf", help="output path without extension")
    p.add_argument("--size", type=float, nargs=2, default=(4.2, 2.6), help="figure size in inches")
    args = p.parse_args()

    dist = lognormal_from_moments(args.mean, args.cov)
    xmax = args.xmax if args.xmax is not None else dist.ppf(0.999)
    x = np.linspace(1e-9, xmax, 1000)
    y = dist.pdf(x)

    fig, ax = plt.subplots(figsize=tuple(args.size))

    if args.fill:
        ax.fill_between(x, y, color="#4C72B0", alpha=0.18, linewidth=0)
    ax.plot(x, y, color="#1F3B73", linewidth=1.4, solid_capstyle="round")

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, y.max() * 1.12)
    ax.set_xlabel(args.label)
    ax.set_ylabel(f"$f_{{{args.label}}}({args.label})$")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(width=0.8, labelsize=9, direction="out")
    ax.set_title(f"Lognormal   $\\mu$ = {args.mean:g},  CoV = {args.cov:g}",
                 fontsize=9, loc="left", pad=8)
    fig.tight_layout()

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("svg", "pdf"):
        fig.savefig(f"{out}.{ext}", transparent=True)
        print(f"wrote {out}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
