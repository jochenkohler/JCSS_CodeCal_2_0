"""Time-variant reliability figure: r(t), s(t) and beta(t) against a target.

Two stacked panels on a common time axis:

  top     resistance realisation r(t), load-effect process s(t) with its
          yearly maxima, and the densities of R and S at t = 0 and at the
          end of the expected service life
  bottom  reliability index beta(t) falling to beta_target at t = t_SL

All text is set with LaTeX (Computer Modern), symbols in math italics.
Output is vector PDF + SVG for Illustrator.

    python plot_reliability_timeline.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gumbel_r, lognorm

matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.size": 9,
    "pdf.fonttype": 42,
    "pdf.compression": 0,
    "svg.fonttype": "path",     # usetex output is vector outlines anyway
    "axes.linewidth": 0.8,
})

C_S = "#B4472E"    # load effect
C_R = "#1F3B73"    # resistance
C_T = "#C0392B"    # target line
GREY = "#333333"
EULER_GAMMA = 0.5772156649015329


def gumbel_from_moments(mean, sd):
    b = sd * np.sqrt(6.0) / np.pi
    return gumbel_r(loc=mean - EULER_GAMMA * b, scale=b)


def lognormal_from_moments(mean, sd):
    cov = sd / mean
    sigma = np.sqrt(np.log(1.0 + cov ** 2))
    return lognorm(s=sigma, scale=np.exp(np.log(mean) - 0.5 * sigma ** 2))


def arrow_axes(ax):
    """Only left/bottom spines, each ending in an arrow head, no ticks."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.plot(1, 0, marker=">", ms=3.2, color=GREY, clip_on=False,
            transform=ax.get_yaxis_transform())
    ax.plot(0, 1, marker="^", ms=3.2, color=GREY, clip_on=False,
            transform=ax.get_xaxis_transform())


def vertical_density(ax, t0, dist, color, width, label=None, lab_dx=0.0):
    """Density laid out sideways on the value axis, anchored on t = t0."""
    y = np.linspace(dist.ppf(2e-4), dist.ppf(1 - 2e-4), 500)
    f = dist.pdf(y)
    x = t0 + width * f / f.max()
    ax.fill_betweenx(y, t0, x, color=color, alpha=0.15, linewidth=0, zorder=3)
    ax.plot(x, y, color=color, lw=1.0, zorder=4)
    ax.plot([t0, t0], [y[0], y[-1]], color=color, lw=0.5, alpha=0.5, zorder=3)
    if label:
        i = int(np.argmax(f))
        ax.text(x[i] + lab_dx, y[i], label, color=color, fontsize=9.5,
                ha="left", va="center", zorder=6)


def load_process(rng, T, mean_max, sd_max, n_per_year=200):
    """Load-effect realisation with a pronounced annual cycle.

    A yearly season of varying intensity plus short-term fluctuation.  The
    series is rescaled so that its yearly maxima carry the prescribed mean
    and standard deviation, i.e. it is consistent with the density of S
    drawn at the two section cuts.
    """
    t = np.linspace(0.0, T, int(T * n_per_year) + 1)

    # annual season, peaked rather than sinusoidal
    season = (0.5 * (1.0 + np.sin(2 * np.pi * (t - 0.30)))) ** 1.15

    # intensity of each season varies from year to year (smoothly interpolated)
    n_years = int(round(T)) + 2
    g_year = np.clip(rng.lognormal(mean=0.0, sigma=0.26, size=n_years), 0.45, 2.2)
    g = np.interp(t, np.arange(n_years) + 0.5, g_year)

    # short-term fluctuation
    k = np.exp(-0.5 * np.linspace(-3, 3, 31) ** 2)
    fast = np.convolve(rng.normal(size=t.size), k / k.sum(), mode="same")
    fast /= fast.std()

    years = np.arange(int(round(T)))

    def rescaled(amp):
        """Process with fluctuation amplitude `amp`, rescaled to the maxima."""
        raw = 0.85 + amp * (0.62 * season * g + 0.115 * fast)
        imax = np.array([np.argmax(np.where((t >= y) & (t < y + 1), raw, -np.inf))
                         for y in years])
        a = sd_max / raw[imax].std()
        return a * raw + (mean_max - a * raw[imax].mean()), imax

    # keep the realisation strictly positive: a load effect never goes below zero
    amp = 1.0
    for _ in range(40):
        s, imax = rescaled(amp)
        if s.min() >= 0.10 * mean_max:
            break
        amp *= 0.92

    return t, s, imax


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsl", type=float, default=20.0, help="expected service life [a]")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--no-design-values", dest="design_values", action="store_false")
    p.add_argument("--out", default="figures/reliability_timeline")
    p.add_argument("--size", type=float, nargs=2, default=(7.2, 4.9))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    T = args.tsl
    tmax = 1.28 * T
    dens_w = 0.085 * T                    # width of the density insets

    # --- models -------------------------------------------------------------
    mR0, sR0 = 2.60, 0.24                 # resistance at t = 0
    mS, sS = 1.00, 0.26                   # yearly maximum of the load effect

    def mR(t):  return mR0 * (1.0 - 0.16 * (t / T) ** 1.8)
    def sR(t):  return sR0 * (1.0 + 0.75 * (t / T) ** 1.5)

    t_s, s_proc, imax = load_process(rng, T, mS, sS)
    t_r = np.linspace(0.0, tmax, 1200)
    r = mR(t_r)

    # --- canvas -------------------------------------------------------------
    fig = plt.figure(figsize=tuple(args.size))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 0.85], hspace=0.34,
                          left=0.09, right=0.99, top=0.94, bottom=0.13)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    y1max = 3.5
    ax1.set_xlim(-0.015 * T, tmax)
    ax1.set_ylim(0, y1max)

    ax2.axvspan(0, T, color="#000000", alpha=0.030, linewidth=0, zorder=0)
    for ax in (ax1, ax2):
        ax.axvline(T, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)

    # --- top panel ----------------------------------------------------------
    # inner 90 % interval of R(t): the trumpet opening with deterioration
    q05 = np.array([lognormal_from_moments(mR(tt), sR(tt)).ppf(0.05) for tt in t_r])
    q95 = np.array([lognormal_from_moments(mR(tt), sR(tt)).ppf(0.95) for tt in t_r])
    ax1.fill_between(t_r, q05, q95, color=C_R, alpha=0.10, linewidth=0, zorder=1)
    for q in (q05, q95):
        ax1.plot(t_r, q, color=C_R, lw=0.6, ls=(0, (4, 3)), alpha=0.55, zorder=1)

    ax1.plot(t_s, s_proc, color=C_S, lw=0.55, alpha=0.85, zorder=2)
    ax1.plot(t_r, r, color=C_R, lw=1.3, zorder=2)
    ax1.plot(t_s[imax], s_proc[imax], "o", ms=2.9, mfc="white", mec=C_S, mew=0.9,
             zorder=5)

    if args.design_values:
        e_d = mS + 2.2 * sS
        r_d = e_d + 0.12
        for lvl, col, lab, va in ((r_d, C_R, r"$r_{\mathrm{d}}$", "bottom"),
                                  (e_d, C_S, r"$e_{\mathrm{d}}$", "top")):
            ax1.plot([0, tmax], [lvl, lvl], color=col, lw=0.6, ls=(0, (5, 4)),
                     alpha=0.65, zorder=1)
            ax1.text(1.17 * T, lvl + (0.04 if va == "bottom" else -0.04), lab,
                     color=col, fontsize=8.5, ha="left", va=va)

    for t0 in (0.0, T):
        first = (t0 == 0.0)
        vertical_density(ax1, t0, lognormal_from_moments(mR(t0), sR(t0)), C_R,
                         dens_w, label=r"$R$" if first else None, lab_dx=0.20)
        vertical_density(ax1, t0, gumbel_from_moments(mS, sS), C_S,
                         dens_w, label=r"$S$" if first else None, lab_dx=0.20)

    ax1.text(0.42 * T, mR(0.42 * T) + 0.30, r"realisation of $r(t)$",
             color=C_R, fontsize=9, ha="center", va="bottom")
    i_q = int(np.argmin(np.abs(t_r - 0.70 * T)))
    ax1.text(t_r[i_q], q95[i_q] + 0.07, r"inner $90\,\%$ interval of $R(t)$",
             color=C_R, fontsize=8.5, ha="center", va="bottom", alpha=0.9)

    win = (t_s > 0.16 * T) & (t_s < 0.24 * T)
    i_lab = int(np.flatnonzero(win)[np.argmin(np.abs(s_proc[win] - 0.8 * mS))])
    ax1.annotate(r"realisation of $s(t)$",
                 xy=(t_s[i_lab], s_proc[i_lab]), xytext=(0.20 * T, 1.72),
                 color=C_S, fontsize=9, ha="center", va="bottom", annotation_clip=False,
                 arrowprops=dict(arrowstyle="-", lw=0.55, color=C_S,
                                 shrinkA=3, shrinkB=2))
    j = imax[9]
    ax1.annotate(r"yearly maxima $s_{\max,i}$",
                 xy=(t_s[j], s_proc[j]), xytext=(0.46 * T, 1.72),
                 color=C_S, fontsize=9, ha="center", va="bottom", annotation_clip=False,
                 arrowprops=dict(arrowstyle="-", lw=0.55, color=C_S,
                                 shrinkA=3, shrinkB=3))

    ax1.text(T, y1max * 0.995, r"$t = t_{\mathrm{SL}}$", fontsize=8.5, color=GREY,
             ha="center", va="top")
    ax1.text(-0.015 * T, y1max, r"$R,\;S$", fontsize=10, ha="right", va="top")
    arrow_axes(ax1)
    # a tick for every year, to make the annual scale readable
    ax1.set_xticks(np.arange(0, int(T) + 1))
    ax1.tick_params(axis="x", length=2.2, width=0.7, color=GREY, labelbottom=False)

    # --- bottom panel -------------------------------------------------------
    tb = np.linspace(0, tmax, 900)
    beta = (mR(tb) - mS) / np.sqrt(sR(tb) ** 2 + sS ** 2)
    beta_t = (mR(T) - mS) / np.sqrt(sR(T) ** 2 + sS ** 2)
    y2max = beta.max() * 1.14

    ax2.plot([0, tmax], [beta_t, beta_t], color=C_T, lw=1.0, ls=(0, (5, 4)),
             zorder=2)
    ax2.plot(tb, beta, color=GREY, lw=1.3, zorder=3)
    ax2.plot([T], [beta_t], "o", ms=3.4, color=C_T, zorder=4)

    ax2.set_ylim(0, y2max)
    ax2.text(0.34 * T, np.interp(0.34 * T, tb, beta) + 0.15, r"$\beta(t)$",
             fontsize=10, ha="center", va="bottom", color=GREY)
    ax2.text(0.02 * T, beta_t + 0.12,
             r"reliability requirement $\beta_{\mathrm{target}}$",
             color=C_T, fontsize=8.5, ha="left", va="bottom")
    ax2.text(-0.02 * T, 0.93 * y2max, r"$\beta$", fontsize=10, ha="right", va="center")
    ax2.set_xlabel(r"time $t$", loc="right", labelpad=0)
    arrow_axes(ax2)

    ybar = -0.085 * y2max
    ax2.annotate("", xy=(0, ybar), xytext=(T, ybar), clip_on=False, zorder=6,
                 arrowprops=dict(arrowstyle="<->,head_width=0.16,head_length=0.4",
                                 lw=0.8, color=GREY, shrinkA=0, shrinkB=0))
    for x in (0.0, T):
        ax2.plot([x, x], [ybar - 0.045 * y2max, ybar + 0.045 * y2max],
                 color=GREY, lw=0.7, clip_on=False, zorder=6)
    ax2.text(0.5 * T, ybar - 0.07 * y2max,
             r"expected service life $t_{\mathrm{SL}} = %g$ a" % T,
             fontsize=8.5, ha="center", va="top", color=GREY, clip_on=False)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{out}.{ext}", transparent=True)
        print(f"wrote {out}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
