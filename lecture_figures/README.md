# Lecture figures

Stand-alone scripts that draw the schematic figures used in the lectures
(densities of R and E, the lognormal density, reliability over time,
reliability updating). They are independent of the `codecal` package and
write vector SVG/PDF files into `figures/`:

    python plot_RE_densities.py
    python plot_reliability_timeline.py      # needs a LaTeX installation (text.usetex)
