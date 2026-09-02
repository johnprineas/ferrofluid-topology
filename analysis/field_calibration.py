"""Fit the electromagnet drive-to-flux calibration used to label every field point.

Reads data/electromagnet_calibration.csv (Hall-probe readings at the dish centre)
and fits B(I). The fit is what converts the coil currents recorded alongside every
spike count into the magnetic flux densities quoted in docs/RESULTS.md.

Usage:
    python analysis/field_calibration.py [--plot assets/figures/field_calibration.png]
"""

import argparse
import csv
import os

import numpy as np


def load_calibration(path):
    """Return (current_mA, voltage_V, B_mT) from the calibration CSV, skipping # comments."""
    current, voltage, field = [], [], []
    with open(path, newline="") as f:
        rows = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in rows:
            current.append(float(row["current_mA"]))
            voltage.append(float(row["voltage_V"]))
            field.append(float(row["B_mT"]))
    return np.array(current), np.array(voltage), np.array(field)


def fit_through_origin(x, y):
    """Least-squares slope for y = m*x, with the standard error on m.

    The coil is driven well below saturation, so B is expected to be proportional
    to I with no offset; forcing the intercept to zero keeps the one free
    parameter physically meaningful (mT per mA).
    """
    m = float(np.sum(x * y) / np.sum(x * x))
    residuals = y - m * x
    dof = len(x) - 1
    s_err = float(np.sqrt(np.sum(residuals**2) / dof / np.sum(x * x)))
    r2 = 1.0 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)
    return m, s_err, float(r2), residuals


def main():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.path.join(here, "data", "electromagnet_calibration.csv"))
    parser.add_argument("--plot", default=None, help="write a PNG of the fit to this path")
    args = parser.parse_args()

    current, voltage, field = load_calibration(args.csv)
    slope, slope_err, r2, residuals = fit_through_origin(current, field)
    resistance = float(np.sum(current * voltage) / np.sum(current * current)) * 1000.0

    print(f"n points              : {len(current)}")
    print(f"Current range         : {current.min():.1f} - {current.max():.1f} mA")
    print(f"Field range           : {field.min():.1f} - {field.max():.1f} mT")
    print(f"Calibration B(I)      : {slope * 1000:.4f} +/- {slope_err * 1000:.4f} mT/A")
    print(f"                        (r^2 = {r2:.5f})")
    print(f"Residual RMS          : {np.sqrt(np.mean(residuals ** 2)):.3f} mT")
    print(f"Coil resistance V(I)  : {resistance:.2f} ohm")
    print()
    for probe in (177.6, 187.96, 208.64, 250.57):
        print(f"  I = {probe:6.2f} mA  ->  B = {slope * probe:5.2f} mT")

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        grid = np.linspace(0, current.max() * 1.05, 200)
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(7.6, 6.4), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}, facecolor="#0d1117",
        )
        for a in (ax, axr):
            a.set_facecolor("#0d1117")
            a.tick_params(colors="#8b949e")
            a.grid(alpha=0.15, color="#8b949e")
            for sp in a.spines.values():
                sp.set_color("#30363d")

        ax.plot(grid, slope * grid, color="#58a6ff", lw=1.6,
                label=f"B = {slope * 1000:.3f} mT/A x I   (r$^2$ = {r2:.4f})")
        ax.scatter(current, field, s=34, color="#f0883e", zorder=3, label="Hall-probe readings")
        ax.set_ylabel("Flux density B (mT)", color="#c9d1d9")
        ax.set_title("Electromagnet calibration at the dish centre", color="#e6edf3", fontsize=13)
        ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=9)

        axr.axhline(0, color="#58a6ff", lw=1)
        axr.scatter(current, residuals, s=26, color="#f0883e", zorder=3)
        axr.set_xlabel("Coil current I (mA)", color="#c9d1d9")
        axr.set_ylabel("Residual (mT)", color="#c9d1d9")

        fig.tight_layout()
        os.makedirs(os.path.dirname(args.plot), exist_ok=True)
        fig.savefig(args.plot, dpi=140, facecolor="#0d1117")
        print(f"\nWrote {args.plot}")


if __name__ == "__main__":
    main()
