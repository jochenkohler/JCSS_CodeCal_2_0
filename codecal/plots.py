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

# ---- the layout of the AHG study's figures -----------------------------------------
# The violin ("bean") plots and the β-against-a_q grids reproduce the figures of
# the 2021 AHG working document and of the Structural Safety / chapter
# manuscripts: statsmodels' beanplot with the options used there (grey violins
# cut at the data range, jittered dots, the mean as a blue bar, the calibrated
# set in teal on the right-hand half), and a 3 × 3 grid of panels per failure
# mode filled column by column, with min–max bands over the (load, a_G) lines
# and the target as a red line.
VIOLIN_GREY = (0.7, 0.7, 0.7)
CALIBRATED = '#009D91'
BEAN_OPTS = {'cutoff': True, 'cutoff_type': 'abs', 'cutoff_val': 0, 'violin_fc': VIOLIN_GREY,
             'label_rotation': 90, 'bean_show_mean': True, 'bean_show_median': False,
             'jitter_marker': '.', 'jitter_marker_size': .5}
BAND_COLORS = ['red', 'blue', 'green', 'orange', 'purple']
SHORT_NAMES = {                           # the labels of the AHG figures
    'Steel bending': 'Steel bend.', 'Concrete compression': 'Concrete comp.', 'Concrete bending': 'Concrete bend.',
    'Glulam bending': 'Glulam bend.', 'Timber bending': 'Timber bend.', 'Masonry': 'Masonry',
    'Aluminium bending': 'Aluminium bend.', 'Pile foundation': 'Pile found.', 'Shallow foundation': 'Shallow found.',
    'wind': 'Wind 50a-extreme', 'snow': 'Snow 50a-extreme', 'imposed': 'Imposed 50a-extreme',
}


def short(name):
    return SHORT_NAMES.get(name, name)


def _fig(w=9, h=4.5):
    fig = Figure(figsize=(w, h))
    return fig, fig.subplots()


def _groups(df, by):
    """[(column, value), …] of the violins: loads and/or failure modes."""
    if by == 'both':
        return [('load', g) for g in dict.fromkeys(df['load'])] + [('mode', g) for g in dict.fromkeys(df['mode'])]
    return [(by, g) for g in dict.fromkeys(df[by])]


def plot_beta_distribution(df, by='both', beta_target=None, ref_df=None,
                           labels=('Calibrated gamma', 'Existing gamma'), title=None, ylim=(2.0, 5.5), T_ref=50):
    """The bean plot of the AHG study: one violin per load and per failure
    mode (`by='both'`, the layout of the study), or per failure mode
    (`'mode'`) or per load (`'load'`); the design situations as jittered
    dots, the weighted mean as a blue bar. With `ref_df` (e.g. the sweep
    before calibration) the reference is drawn in grey on the left-hand
    half and `df` in teal on the right ("Effect of calibration"). `df` may
    also be a dict {label: sweep} — one violin per sweep, as in the
    comparison of the design equations (8.12), (8.13), (8.14).
    `beta_target` is accepted for API compatibility and not drawn, as in
    the original figures."""
    import statsmodels.api as sm
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(*ylim)
    ax.grid(axis='y')
    if isinstance(df, dict):
        names = list(df)
        box = [frame['beta'].to_numpy() for frame in df.values()]
        sm.graphics.beanplot(box, ax=ax, labels=names, side='both', jitter=True, plot_opts=dict(BEAN_OPTS))
        ax.set_title(title or 'Reliability Benchmark Design Equations')
    else:
        groups = _groups(df, by)
        names = [short(g) for _, g in groups]
        box = [df.loc[df[k] == g, 'beta'].to_numpy() for k, g in groups]
        if ref_df is None:
            sm.graphics.beanplot(box, ax=ax, labels=names, side='both', jitter=True, plot_opts=dict(BEAN_OPTS))
            ax.set_title(title or 'Reliability Benchmark')
        else:
            box_ref = [ref_df.loc[ref_df[k] == g, 'beta'].to_numpy() for k, g in groups]
            opts = dict(BEAN_OPTS, bean_legend_text=labels[1])
            sm.graphics.beanplot(box_ref, ax=ax, labels=names, side='left', jitter=True, plot_opts=opts)
            opts = dict(BEAN_OPTS, violin_fc=CALIBRATED, bean_color=CALIBRATED, bean_mean_color=CALIBRATED,
                        bean_legend_text=labels[0])
            sm.graphics.beanplot(box, ax=ax, labels=names, side='right', jitter=True, plot_opts=opts)
            ax.set_title(title or 'Effect of calibration')
    ax.set_ylabel(r'$\beta_{' + f'{T_ref:g}' + 'a}$')
    return fig


def plot_beta_vs_aQ(df, beta_target=3.8, ref_df=None, title=None, ylim=(2.0, 5.5), T_ref=50):
    """The 3 × 3 grid of the AHG study: one panel per failure mode, filled
    column by column, with β against the variable-load share a_q — one thin
    black line per (load, a_G) and the min–max band over them in red; the
    target as a red line. With `ref_df` the reference band is drawn in light
    red behind the band of `df` (blue), as in the calibration figures. `df`
    may be a dict {label: sweep}: one band per sweep in red / blue / green,
    as in the comparison of the design equations."""
    frames = df if isinstance(df, dict) else {'current': df}
    first = next(iter(frames.values()))
    modes = list(dict.fromkeys(first['mode']))
    n = len(modes)
    nrow = min(3, n)
    ncol = int(np.ceil(n / nrow))
    fig = Figure(figsize=(10, 10) if (nrow, ncol) == (3, 3) else (3.4 * ncol, 3.4 * nrow))
    axes = fig.subplots(nrow, ncol, sharex=True, sharey=True, squeeze=False)

    def piv(frame, mode):
        sub = frame[frame['mode'] == mode]
        return sub.pivot_table(index='a_Q', columns=['load', 'a_G'], values='beta')

    for i, mode in enumerate(modes):
        ii, jj = i % nrow, i // nrow                      # column by column, as in the study
        ax = axes[ii, jj]
        if isinstance(df, dict):
            for color, (label, frame) in zip(BAND_COLORS, frames.items()):
                p = piv(frame, mode)
                ax.fill_between(p.index, p.max(axis=1), p.min(axis=1), alpha=.4, linewidth=0, color=color)
        elif ref_df is not None:
            p_ref, p = piv(ref_df, mode), piv(df, mode)
            ax.fill_between(p_ref.index, p_ref.max(axis=1), p_ref.min(axis=1), alpha=.2, linewidth=0, color='red')
            ax.plot(p.index, p.to_numpy(), linewidth=0.5, color='black')
            ax.fill_between(p.index, p.max(axis=1), p.min(axis=1), alpha=.5, linewidth=0)
        else:
            p = piv(df, mode)
            ax.fill_between(p.index, p.max(axis=1), p.min(axis=1), alpha=.4, linewidth=0, color='red')
            ax.plot(p.index, p.to_numpy(), linewidth=0.5, color='black')
        if beta_target is not None:
            ax.plot(np.linspace(0, 1, 10), np.ones(10) * beta_target, linewidth=2, color='red')
        ax.set_title(short(mode))
        ax.grid()
        ax.set(xlim=(0, 1), ylim=ylim)
        if jj == 0:
            ax.set_ylabel(r'$\beta^{' + f'{T_ref:g}' + 'a}$')
        if ii == nrow - 1:
            ax.set_xlabel(r'$a_q$')
    for j in range(n, nrow * ncol):
        axes[j % nrow, j // nrow].set_visible(False)
    if title:
        fig.suptitle(title)
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
