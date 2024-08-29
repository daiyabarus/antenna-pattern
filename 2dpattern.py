import matplotlib.pyplot as plt
import numpy as np


def plot_2D_Pattern_polar_dB(
    angdeg, rdB, rangedB=(-30, 10), stepdB=10, stepdeg=30, noLeg=False
):
    """
    Plots 2D polar plot of angle-vs-r in logarithmic scale (for r).

    Parameters
    ----------
        angdeg : numpy.ndarray
            Polar angles in degrees, expected to be in the range [0, 360].
        rdB : numpy.ndarray
            Radial distance in dB (logarithmic scale).
        rangedB : tuple, optional
            Min and max dB range for the plot, default is [-30, +10] dB.
        stepdB : int, optional
            dB step for the plot markings, default is 10 dB.
        stepdeg : int, optional
            Degree step for the plot markings, default is 30 degrees.
        noLeg : bool, optional
            If True, disables dB marks on the right side, default is False.
    """
    # Input checks and defaults
    if len(angdeg) != len(rdB):
        raise ValueError("angdeg and rdB must be of equal length!")

    if rangedB[1] <= rangedB[0]:
        raise ValueError("rangedB should be [min_dB, max_dB]!")

    if np.any(np.iscomplex(rdB)):
        raise ValueError("rdB should be real-valued!")

    # Prepare plot area
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})

    # Normalize rdB to fit within the rangedB limits
    rdBcr = np.clip(rdB, rangedB[0], rangedB[1])
    rdBsc = 1 + (rdBcr - rangedB[1]) / (rangedB[0] - rangedB[1])

    # Plot the patterns
    ax.plot(np.radians(angdeg), rdBsc, linewidth=2)

    if max(angdeg) <= 180:
        ax.plot(np.radians(angdeg + 180), rdBsc, linestyle=":", linewidth=1)

    # Plot cosmetics: iso-dB circles
    c_log = np.arange(rangedB[0], rangedB[1] + stepdB, stepdB)
    c = 1 + (c_log - rangedB[1]) / (rangedB[0] - rangedB[1])

    for k in range(1, len(c) - 1):
        ax.plot(
            np.linspace(0, 2 * np.pi, 100),
            np.ones(100) * c[k],
            linestyle=":",
            color="black",
        )

    # Markers/labels for the iso-radial (dB) distance circles
    if not noLeg:
        for k, dBval in enumerate(reversed(c_log)):
            ax.text(
                0, c[k] + 0.02, f"{dBval:+.0f} dB", va="center", ha="left", fontsize=10
            )

    # Rays -- Indicating the angles
    rays = 360 // stepdeg
    phi_s = np.linspace(0, 2 * np.pi, rays + 1)
    for phi in phi_s:
        ax.plot([phi, phi], [0, 1], linestyle=":", color="black")
        ax.text(
            phi, 1.1, f"{np.degrees(phi):.0f}°", ha="center", va="center", fontsize=10
        )

    # Final touches
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.grid(False)
    ax.set_axis_off()
    plt.show()


# Test the function with a sample input
if __name__ == "__main__":
    theta = np.linspace(np.finfo(float).eps, np.pi, 100)
    kLs = 2 * np.pi * np.array([0.001, 0.5, 1, 1.5])
    r = np.zeros((len(kLs), len(theta)))

    for k, kL in enumerate(kLs):
        F = ((np.cos(kL / 2 * np.cos(theta)) - np.cos(kL / 2)) / np.sin(theta)) ** 2
        r[k, :] = 2 * F / np.trapezoid(F * np.sin(theta), theta)

    # Replace zeros or very small values in r to avoid log10 of zero or negative
    r[r <= 0] = np.finfo(float).eps

    rdB = 10 * np.log10(r)
    angdeg = np.degrees(theta)

    plot_2D_Pattern_polar_dB(angdeg, rdB[0])
