#!/usr/bin/env python3
"""
Run a calibration of the AHG model from the command line.

    python run_calibration.py                                # eq. 8.12, γ_G and γ_Q, FORM
    python run_calibration.py --equation 8.13 --free xi gamma_G gamma_Q
    python run_calibration.py --free gamma_G gamma_Q:wind gamma_Q:snow gamma_Q:imposed
    python run_calibration.py --method sorm --target 4.3 --out results_cc3

Writes <out>/status_quo.csv, <out>/calibrated.csv (one row per design
situation), <out>/summary.txt and the before/after figures.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codecal as cc                      # noqa: E402
from codecal import plots                 # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--equation', default='8.12', choices=cc.model.EQUATIONS)
    p.add_argument('--free', nargs='+', default=['gamma_G', 'gamma_Q'],
                   help="parameters to calibrate: gamma_G gamma_GP gamma_Q gamma_Q:<load> xi")
    p.add_argument('--method', default='form', choices=['form', 'sorm'])
    p.add_argument('--target', type=float, default=3.8, help='target reliability index')
    p.add_argument('--T-ref', type=float, default=50.0, help='reference period in years')
    p.add_argument('--n-aQ', type=int, default=10)
    p.add_argument('--n-aG', type=int, default=3)
    p.add_argument('--seed', type=int, default=cc.ahg2021.DEFAULT_SEED)
    p.add_argument('--out', default='results')
    a = p.parse_args(argv)

    os.makedirs(a.out, exist_ok=True)
    model = cc.ahg_model(T_ref=a.T_ref, n_aQ=a.n_aQ, n_aG=a.n_aG, beta_target=a.target, seed=a.seed)
    print(model.summary())
    print(f"\ncalibrating {a.free} for eq. {a.equation} with {a.method.upper()}, β_t = {a.target}\n")
    t0 = time.time()
    res = cc.calibrate(model, free=tuple(a.free), equation=a.equation, method=a.method,
                       start=cc.START_VALUES, beta_target=a.target)
    print(f"\n{res}\n({time.time() - t0:.0f} s)")

    res.sweep_before.to_csv(os.path.join(a.out, 'status_quo.csv'), index=False)
    res.sweep_after.to_csv(os.path.join(a.out, 'calibrated.csv'), index=False)
    with open(os.path.join(a.out, 'summary.txt'), 'w') as f:
        f.write(f"equation {a.equation}, method {a.method}, T_ref {a.T_ref}, seed {a.seed}\n")
        f.write(str(res) + '\n\nmean beta per mode x load, calibrated:\n')
        f.write(cc.beta_table(res.sweep_after).round(3).to_string() + '\n')
    for name, fig in [
        ('beta_by_mode.png', plots.plot_beta_distribution(res.sweep_after, 'mode', a.target, res.sweep_before,
                                                          labels=('calibrated', 'start values'))),
        ('beta_by_load.png', plots.plot_beta_distribution(res.sweep_after, 'load', a.target, res.sweep_before,
                                                          labels=('calibrated', 'start values'))),
        ('beta_vs_aQ.png', plots.plot_beta_vs_aQ(res.sweep_after, a.target, res.sweep_before)),
        ('sensitivities.png', plots.plot_sensitivities(res.sweep_after)),
        ('history.png', plots.plot_calibration_history(res)),
    ]:
        fig.savefig(os.path.join(a.out, name), dpi=150, bbox_inches='tight')
    print(f"results written to {a.out}/")
    return res


if __name__ == '__main__':
    main()
