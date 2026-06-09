"""Matplotlib defaults aligned with the M2PI LaTeX report typography.

The project report uses ``\\documentclass[11pt]{m2pi}`` without an explicit
font package, so body text is set in Computer Modern Roman. Matplotlib ships
compatible Computer Modern fonts, which keeps plots portable without requiring
a local LaTeX installation.
"""

from __future__ import annotations

import matplotlib as mpl

M2PI_RCPARAMS = {
    "font.family": "serif",
    "font.serif": [
        "Computer Modern Roman",
        "CMU Serif",
        "Latin Modern Roman",
        "DejaVu Serif",
    ],
    "font.sans-serif": [
        "Computer Modern Sans Serif",
        "CMU Sans Serif",
        "Latin Modern Sans",
        "DejaVu Sans",
    ],
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
}


def apply_m2pi_style() -> None:
    """Apply report typography to matplotlib's global rcParams."""
    mpl.rcParams.update(M2PI_RCPARAMS)


apply_m2pi_style()
