"""Companion figure: new information on the resistance extends the service life.

Same two-panel layout as plot_reliability_timeline.py.  At t_I information
on R becomes available (proof loading, inspection, measurement).  The
resistance distribution is updated - less scatter, slightly higher mean -
the trumpet narrows, beta(t) jumps up and reaches beta_target much later,
which is the extension of the expected service life.

    python plot_reliability_updating.py
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_reliability_timeline import (C_R, C_S, C_T, GREY, arrow_axes,
                                       gumbel_from_moments, load_process,
                                       lognormal_from_moments, vertical_density)

C_U = "#1B7A6B"    # updated resistance


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tsl", type=float, default=20.0, help="service life before updating [a]")
    p.add_argument("--ti", type=float, default=10.0, help="time of the new information [a]")
    p.add_argument("--sd-ratio", type=float, default=0.45,
                   help="posterior/prior standard deviation of R at t_I")
    p.add_argument("--mean-shift", type=float, default=0.14,
                   help="shift of the mean of R at t_I from the new information")
    p.add_argument("--sd-growth", type=float, default=0.75,
                   help="growth of sigma_R over the service life (trumpet opening)")
    p.add_argument("--deterioration", type=float, default=0.16,
                   help="relative loss of mean resistance over the service life")
    p.add_argument("--seed", type=int, default=12)
    p.add_argument("--out", default="figures/reliability_updating")
    p.add_argument("--size", type=float, nargs=2, default=(7.2, 4.9))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    T, tI = args.tsl, args.ti

    # --- models (prior identical to the first figure) ------------------------
    mR0, sR0 = 2.60, 0.24
    mS, sS = 1.00, 0.26

    def mR(t):  return mR0 * (1.0 - args.deterioration * (t / T) ** 1.8)
    def sR(t):  return sR0 * (1.0 + args.sd_growth * (t / T) ** 1.5)

    # updating at t_I: same relative deterioration afterwards
    f_m = (mR(tI) + args.mean_shift) / mR(tI)
    f_s = args.sd_ratio

    def mU(t):  return f_m * mR(t)
    def sU(t):  return f_s * sR(t)

    def beta(mean, sd, t):
        return (mean(t) - mS) / np.sqrt(sd(t) ** 2 + sS ** 2)

    beta_t = beta(mR, sR, T)                       # requirement, as before

    # extended service life: where the updated beta reaches the same target
    tt = np.linspace(tI, 6 * T, 20000)
    b_u = beta(mU, sU, tt)
    T_new = float(tt[np.argmax(b_u <= beta_t)])
    tmax = T_new * 1.16

    t_s, s_proc, imax = load_process(rng, T_new, mS, sS)
    t_r = np.linspace(0.0, tmax, 1600)
    t_u = np.linspace(tI, tmax, 1200)

    # --- canvas -------------------------------------------------------------
    fig = plt.figure(figsize=tuple(args.size))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.55, 0.85], hspace=0.34,
                          left=0.09, right=0.99, top=0.94, bottom=0.13)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    y1max = 3.5
    dens_w = 0.085 * T_new
    ax1.set_xlim(-0.015 * T_new, tmax)
    ax1.set_ylim(0, y1max)

    for ax in (ax1, ax2):
        ax.axvline(tI, color=C_U, lw=0.9, ls=(0, (3, 2)), zorder=1)
        ax.axvline(T, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1, alpha=0.8)
        ax.axvline(T_new, color=GREY, lw=0.7, ls=(0, (4, 3)), zorder=1)
    ax2.axvspan(0, T, color="#000000", alpha=0.030, linewidth=0, zorder=0)
    ax2.axvspan(T, T_new, color=C_U, alpha=0.055, linewidth=0, zorder=0)

    # --- top panel: prior and updated trumpets ------------------------------
    def band(ax, tv, mean, sd, color, alpha, dashed_edges=True):
        q05 = np.array([lognormal_from_moments(mean(x), sd(x)).ppf(0.05) for x in tv])
        q95 = np.array([lognormal_from_moments(mean(x), sd(x)).ppf(0.95) for x in tv])
        ax.fill_between(tv, q05, q95, color=color, alpha=alpha, linewidth=0, zorder=1)
        if dashed_edges:
            for q in (q05, q95):
                ax.plot(tv, q, color=color, lw=0.6, ls=(0, (4, 3)), alpha=0.55,
                        zorder=1)
        return q05, q95

    q05p, q95p = band(ax1, t_r, mR, sR, C_R, 0.09)
    q05u, q95u = band(ax1, t_u, mU, sU, C_U, 0.13)

    ax1.plot(t_s, s_proc, color=C_S, lw=0.5, alpha=0.8, zorder=2)
    ax1.plot(t_s[imax], s_proc[imax], "o", ms=2.6, mfc="white", mec=C_S, mew=0.85,
             zorder=5)
    ax1.plot(t_r, mR(t_r), color=C_R, lw=1.2, zorder=2)
    ax1.plot(t_u, mU(t_u), color=C_U, lw=1.4, zorder=3)

    # densities: prior and updated at t_I, updated at the new end of life
    vertical_density(ax1, 0.0, lognormal_from_moments(mR(0), sR(0)), C_R, dens_w,
                     label=r"$R$", lab_dx=0.25)
    vertical_density(ax1, 0.0, gumbel_from_moments(mS, sS), C_S, dens_w,
                     label=r"$S$", lab_dx=0.25)
    vertical_density(ax1, tI, lognormal_from_moments(mR(tI), sR(tI)), C_R, dens_w)
    vertical_density(ax1, tI, lognormal_from_moments(mU(tI), sU(tI)), C_U, dens_w)
    vertical_density(ax1, T_new, lognormal_from_moments(mU(T_new), sU(T_new)),
                     C_U, dens_w)
    vertical_density(ax1, T_new, gumbel_from_moments(mS, sS), C_S, dens_w)

    ax1.annotate(r"new information on $R$", xy=(tI, 3.10), xytext=(tI - 1.2, 3.34),
                 color=C_U, fontsize=9, ha="right", va="bottom",
                 annotation_clip=False,
                 arrowprops=dict(arrowstyle="->,head_width=0.13,head_length=0.35",
                                 lw=0.7, color=C_U, shrinkA=2, shrinkB=2))
    x_u = tI + 0.55 * (T_new - tI)
    ax1.text(x_u, np.interp(x_u, t_u, q95u) - 0.12, r"updated $R \mid I$",
             color=C_U, fontsize=8.5, ha="center", va="top")
    x_p = 0.22 * T_new
    ax1.text(x_p, np.interp(x_p, t_r, q95p) - 0.12, r"prior $R(t)$", color=C_R,
             fontsize=8.5, ha="center", va="top")
    ax1.text(T_new, y1max * 0.995, r"$t_{\mathrm{SL}}^{*}$", fontsize=8.5,
             color=GREY, ha="center", va="top")
    ax1.text(T, y1max * 0.995, r"$t_{\mathrm{SL}}$", fontsize=8.5, color=GREY,
             ha="center", va="top", alpha=0.8)
    ax1.text(0.16 * T_new, 1.55, r"realisation of $s(t)$", color=C_S, fontsize=9,
             ha="center", va="bottom")
    ax1.text(-0.015 * T_new, y1max, r"$R,\;S$", fontsize=10, ha="right", va="top")
    arrow_axes(ax1)
    ax1.set_xticks(np.arange(0, int(tmax) + 1))
    ax1.tick_params(axis="x", length=2.2, width=0.7, color=GREY, labelbottom=False)

    # --- bottom panel: the two reliability histories ------------------------
    tb = np.linspace(0, tmax, 1200)
    b_prior = beta(mR, sR, tb)
    b_upd = beta(mU, sU, t_u)
    y2max = max(b_prior.max(), b_upd.max()) * 1.16

    ax2.plot([0, tmax], [beta_t, beta_t], color=C_T, lw=1.0, ls=(0, (5, 4)),
             zorder=2)
    ax2.plot(tb, b_prior, color=GREY, lw=1.0, ls=(0, (4, 2.5)), alpha=0.75, zorder=3)
    ax2.plot(tb[tb <= tI], b_prior[tb <= tI], color=GREY, lw=1.3, ls="-", zorder=3)
    ax2.plot(t_u, b_upd, color=C_U, lw=1.4, zorder=4)
    ax2.plot([tI, tI], [beta(mR, sR, tI), b_upd[0]], color=C_U, lw=1.0,
             ls=(0, (1.5, 1.5)), zorder=4)
    ax2.plot([T, T_new], [beta_t, beta_t], "o", ms=3.4, ls="none", color=C_T,
             zorder=5)

    ax2.set_ylim(0, y2max)
    ax2.text(0.30 * T, np.interp(0.30 * T, tb, b_prior) + 0.16, r"$\beta(t)$",
             fontsize=10, ha="center", va="bottom", color=GREY)
    x_b = tI + 0.35 * (T_new - tI)
    ax2.text(x_b, np.interp(x_b, t_u, b_upd) + 0.16,
             r"$\beta(t \mid I)$", fontsize=10, ha="center", va="bottom", color=C_U)
    ax2.text(0.02 * T, beta_t + 0.12, r"$\beta_{\mathrm{target}}$", color=C_T,
             fontsize=9, ha="left", va="bottom")
    ax2.text(-0.02 * T_new, 0.93 * y2max, r"$\beta$", fontsize=10, ha="right",
             va="center")
    ax2.set_xlabel(r"time $t$", loc="right", labelpad=0)
    arrow_axes(ax2)

    # service life and its extension
    ybar = -0.085 * y2max
    labels = ((0.0, T, GREY,
               r"expected service life $t_{\mathrm{SL}} = %g$ a" % T),
              (T, T_new, C_U,
               r"extension $\Delta t_{\mathrm{SL}} \approx %g$ a" % round(T_new - T)))
    for x0, x1, col, lab in labels:
        ax2.annotate("", xy=(x0, ybar), xytext=(x1, ybar), clip_on=False, zorder=6,
                     arrowprops=dict(arrowstyle="<->,head_width=0.16,head_length=0.4",
                                     lw=0.8, color=col, shrinkA=0, shrinkB=0))
        for x in (x0, x1):
            ax2.plot([x, x], [ybar - 0.045 * y2max, ybar + 0.045 * y2max],
                     color=col, lw=0.7, clip_on=False, zorder=6)
        ax2.text(0.5 * (x0 + x1), ybar - 0.07 * y2max, lab, fontsize=8.5,
                 ha="center", va="top", color=col, clip_on=False)

    print(f"beta_target = {beta_t:.2f},  t_SL = {T:g} a,  t_SL* = {T_new:.1f} a")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    for ext in ("pdf", "svg"):
        fig.savefig(f"{out}.{ext}", transparent=True)
        print(f"wrote {out}.{ext}")
    plt.close(fig)


if __name__ == "__main__":
    main()
