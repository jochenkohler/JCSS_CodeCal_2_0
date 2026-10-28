"""
codecal — reliability-based calibration of partial factors, written to be read.

    distributions   Normal / Lognormal / Gumbel from mean and CoV; u → x transform
    variables       RandomVariable with characteristic value; time-variant loads
    reliability     FORM (HL-RF), SORM (Breitung), Monte Carlo, importance sampling
    model           FailureMode, VariableLoad, DesignSituation, design equations,
                    the sweep over design situations (evaluate)
    calibrate       the objective Σ w (β − β_t)² and its SLSQP minimisation
    ahg2021         the AHG "Reliability Background of the Eurocodes" model as data
    plots           figures for the lessons and reports

Start with lessons/01_random_variables.py.
"""

from .ahg2021 import START_VALUES, ahg_model
from .calibrate import CalibrationResult, calibrate, objective
from .distributions import Gumbel, Lognormal, Normal, distribution
from .model import (CalibrationModel, DesignSituation, FailureMode, PartialFactors,
                    VariableLoad, beta_table, evaluate, weighted_mean_beta)
from .reliability import (FORMResult, LimitState, form, importance_sampling,
                          monte_carlo, sorm)
from .variables import RandomVariable, rv, time_variant_load

__version__ = '2.0'
