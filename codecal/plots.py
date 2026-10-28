"""
Figures for the lessons and for calibration reports (matplotlib only).

All functions return a matplotlib Figure (created without pyplot's figure
manager, so they work in scripts, notebooks and web apps alike) — call
`fig.savefig(...)` or let the notebook display it.
"""

from __future__ import annotations

import io

import matplotlib
from matplotlib.figure import Figure
import numpy as np

from .model import beta_table


def figure_png(fig, dpi=110):
    """PNG bytes of a Figure (used for inline display in IPython/Jupyter)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight')
    return buf.getvalue()


def _register_inline_display():
    """When running under IPython (Jupyter, Spyder), show a Figure that is
    the last expression of a cell as an image — the Figures made here do
    not go through pyplot, so the usual inline backend never sees them."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            ip.display_formatter.formatters['image/png'].for_type(Figure, figure_png)
            ip.display_formatter.formatters['text/plain'].for_type(
                Figure, lambda fig, p, cycle: p.text(''))
    except Exception:            # no IPython, or an unexpected one — plain scripts don't care
        pass


_register_inline_display()

CURRENT = '#1f5fa8'
REFERENCE = '#c0504d'
TARGET = '#d62728'
GREY = '#777777'


def _fig(w=9, h=4.5):
    fig = Figure(figsize=(w, h))
    return fig, fig.subplots()


def plot_beta_distribution(df, by='mode', beta_target=None, ref_df=None,
                           labels=('current', 'reference'), title=None, ylim=(2.0, 5.5)):
    """Spread of β per failure mode (or per load): a violin per group with
    the individual design situations as dots and the weighted mean as a
    bar. With `ref_df`, the reference is drawn in red to the left."""
    groups = list(dict.fromkeys(df[by]))
    fig, ax = _fig(max(7, 0.9 * len(groups) + 2), 4.5)
    pos = np.arange(len(groups))

    def draw(frame, offset, color, label):
        data = [frame.loc[frame[by] == g, 'beta'].to_numpy() for g in groups]
        parts = ax.violinplot(data, positions=pos + offset, widths=0.38, showextrema=False)
        for body in parts['bodies']:
            body.set_facecolor(color)
            body.set_alpha(0.25)
        rng = np.random.default_rng(0)
        for p, d in zip(pos + offset, data):
            ax.plot(p + rng.uniform(-0.08, 0.08, len(d)), d, '.', color=color, ms=3, alpha=0.6)
            w = frame.loc[frame[by] == groups[int(round(p - offset))], 'weight'].to_numpy()
            ax.plot([p - 0.17, p + 0.17], [np.average(d, weights=w)] * 2, color=color, lw=2,
                    label=label if p == pos[0] + offset else None)

    if ref_df is not None:
        draw(ref_df, -0.21, REFERENCE, labels[1])
        draw(df, 0.21, CURRENT, labels[0])
    else:
        draw(df, 0.0, CURRENT, labels[0])
    if beta_target is not None:
        ax.axhline(beta_target, color=TARGET, lw=1.2, ls='--', label=f'β_t = {beta_target}')
    ax.set_xticks(pos)
    ax.set_xticklabels(groups, rotation=30, ha='right')
    ax.set_ylim(*ylim)
    ax.set_ylabel('β')
    ax.grid(axis='y', alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_beta_vs_aQ(df, beta_target=None, ref_df=None, title=None, ylim=(2.0, 5.5)):
    """One panel per failure mode: β against the variable-load share a_Q,
    one line per (load, a_G); shaded band = min/max over them. A
    reference sweep (e.g. before calibration) is shown as a red band."""
    modes = list(dict.fromkeys(df['mode']))
    ncol = 3
    nrow = int(np.ceil(len(modes) / ncol))
    fig = Figure(figsize=(11, 3.2 * nrow))
    axes = fig.subplots(nrow, ncol, sharex=True, sharey=True, squeeze=False)

    def band(ax, frame, color, alpha, label):
        sub = frame[frame['mode'] == mode]
        piv = sub.pivot_table(index='a_Q', columns=['load', 'a_G'], values='beta')
        ax.fill_between(piv.index, piv.min(axis=1), piv.max(axis=1), color=color, alpha=alpha,
                        lw=0, label=label)
        return piv

    for i, mode in enumerate(modes):
        ax = axes[i // ncol, i % ncol]
        if ref_df is not None:
            band(ax, ref_df, REFERENCE, 0.25, 'reference')
        piv = band(ax, df, CURRENT, 0.25, 'current')
        for (load, a_G), col in piv.items():
            ax.plot(piv.index, col, color='k', lw=0.6)
        if beta_target is not None:
            ax.axhline(beta_target, color=TARGET, lw=1.2)
        ax.set_title(mode, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
        if i % ncol == 0:
            ax.set_ylabel('β')
        if i // ncol == nrow - 1:
            ax.set_xlabel('$a_Q$')
    for j in range(len(modes), nrow * ncol):
        axes[j // ncol, j % ncol].set_visible(False)
    axes[0, 0].legend(fontsize=8, loc='lower left')
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_sensitivities(df, by='mode', title='Mean sensitivity factors α² per variable'):
    """Stacked bars of the mean α_i² per group — which variables drive the
    reliability of each failure mode (or load)."""
    cols = [c for c in df.columns if c.startswith('alpha_')]
    groups = list(dict.fromkeys(df[by]))
    fig, ax = _fig(max(7, 0.9 * len(groups) + 2), 4.5)
    bottom = np.zeros(len(groups))
    cmap = matplotlib.colormaps['tab10']
    for k, c in enumerate(cols):
        vals = np.array([(df.loc[df[by] == g, c] ** 2).mean() for g in groups])
        ax.bar(groups, vals, bottom=bottom, label=c.replace('alpha_', ''), color=cmap(k))
        bottom += vals
    ax.set_ylabel('α²  (share of the uncertainty at the design point)')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc='center left', bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=30, ha='right')
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_form_history(res, title='FORM (HL-RF) iterations'):
    """β_k and g(u_k) over the iterations of one FORM run."""
    betas = [h[0] for h in res.history]
    gs = [h[1] for h in res.history]
    fig = Figure(figsize=(8, 3.5))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.plot(range(1, len(betas) + 1), betas, 'o-', color=CURRENT)
    ax1.axhline(res.beta, color=GREY, ls='--')
    ax1.set_xlabel('iteration')
    ax1.set_ylabel('β_k')
    ax1.grid(alpha=0.3)
    ax2.semilogy(range(1, len(gs) + 1), np.abs(gs) + 1e-16, 'o-', color=REFERENCE)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('|g(u_k)|')
    ax2.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_limit_state_2d(ls, res=None, lim=5.0, n=200, title=None):
    """Contours of a two-variable limit state in u-space, the failure
    domain, the design point and the circle of radius β."""
    u1 = np.linspace(-lim, lim, n)
    U1, U2 = np.meshgrid(u1, u1)
    G = np.vectorize(lambda a, b: ls.g_u(np.array([a, b])))(U1, U2)
    fig, ax = _fig(5.5, 5.5)
    ax.contourf(U1, U2, G <= 0, levels=[0.5, 1.5], colors=[REFERENCE], alpha=0.15)
    cs = ax.contour(U1, U2, G, levels=12, colors=GREY, linewidths=0.6)
    ax.clabel(cs, fontsize=7, fmt='%.1f')
    ax.contour(U1, U2, G, levels=[0.0], colors='k', linewidths=1.5)
    if res is not None:
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(res.beta * np.cos(th), res.beta * np.sin(th), '--', color=CURRENT, lw=1)
        ax.plot([0, res.u_star[0]], [0, res.u_star[1]], '-', color=CURRENT, lw=1.5)
        ax.plot(*res.u_star, 'o', color=CURRENT, ms=7, label=f'design point, β = {res.beta:.3f}')
        ax.legend(loc='upper left', fontsize=9)
    ax.plot(0, 0, 'k+', ms=10)
    ax.set_xlabel(f'$u_1$  ({ls.variables[0].name})')
    ax.set_ylabel(f'$u_2$  ({ls.variables[1].name})')
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_calibration_history(result, title='Calibration: objective per evaluation'):
    vals = [h[1] for h in result.history]
    fig, ax = _fig(7, 3.5)
    ax.semilogy(range(1, len(vals) + 1), vals, 'o-', color=CURRENT, ms=4)
    ax.set_xlabel('objective evaluation')
    ax.set_ylabel('M(γ) = Σ w (β − β_t)²')
    ax.grid(alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_gamma_comparison(results: dict, title='Calibrated partial factors'):
    """Dot plot of the calibrated factors of several calibrations
    ({label: CalibrationResult})."""
    names = list(dict.fromkeys(n for r in results.values() for n in r.free))
    fig = Figure(figsize=(2.2 * len(names) + 1, 0.6 * len(results) + 1.5))
    axes = fig.subplots(1, len(names), sharey=True, squeeze=False)[0]
    labels = list(results)
    cmap = matplotlib.colormaps['tab10']
    for ax, name in zip(axes, names):
        for i, (lab, r) in enumerate(results.items()):
            if name in r.free:
                v = r.factors.get(name)
                ax.plot(v, i, 'o', color=cmap(i), ms=8)
                ax.annotate(f'{v:.3f}', (v, i), xytext=(6, 0), textcoords='offset points', va='center', fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_title(name, fontsize=10)
        ax.grid(axis='x', alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
