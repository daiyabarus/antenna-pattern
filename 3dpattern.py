import matplotlib.pyplot as plt
import numpy as np


class Plot3DParams:
    def __init__(self):
        # General stuff
        self.dB_Lim = 40
        self.ra = 1.1  # length of normalized x/y/z-axis vector
        self.cbo = 0  # colorbar on ?
        self.View = [50, 30]  # view [el,az] angles
        self.ScaleColors = 1  # scale colors for uniformity across |E|, P, dB?

        # Spherical-coord-system params
        self.Sph = {
            "xyz": 1,  # plot xyz axes ?
            "Sphere": 0.2,  # plot Sphere ? 0=No, >0: Boldness (0.1=Barely Vis)
            "Circles": 1,  # plot cut-circles ?
        }

        # Cylindrical-coord-system params
        self.Cyl = {
            "Flat_or_Surf": 2,  # 1=flat (pcolor-like), 2=surf
            "Circles_degStep": 15,  # deg-step for concentric-circles
            "Rays_degStep": 30,  # deg-step for radial-rays
            "Circles": {"Annotate": 1},  # annotate theta=90 ?
            "Rays": {"Annotate": 1},  # annotate phi=0 ?
            "xyz": 1,  # plot xy axes ?
            "CapSideWalls": 1,  # plot "cap" and "SWs" of cylinder ?
            "gridCol": 0.2 * np.array([1, 1, 1]),  # cyl-grid line-color
        }


plot3Dparams = Plot3DParams()


def plot_3D_Cylindrical(t, p, r, PT):
    gridCol = plot3Dparams.Cyl["gridCol"]

    # Preps
    ro = r.copy()  # store originally supplied r (needed for color-scaling)
    if (PT - 3) == 3:  # Check for dB and rescale in [0,1], so that it's plottable:
        R = plot3Dparams.dB_Lim
        r = (np.maximum(r, -R) + R) / R

    # Convert from cylindrical (t, p, r) to cartesian (x, y, z) coordinates:
    rhoc = t  # [theta]
    phic = p  # [phi]
    x = rhoc * np.cos(np.radians(phic))
    y = rhoc * np.sin(np.radians(phic))
    z = r

    # If color-scaling was requested
    if plot3Dparams.ScaleColors == 1 and plot3Dparams.Cyl["Flat_or_Surf"] == 2:  # Surf
        if (PT - 3) == 3:  # dB
            col_scale = 10.0 ** (ro / 10)
        elif (PT - 3) == 1:  # |E|
            col_scale = ro**2
        else:
            col_scale = r
    else:  # Unscaled or Flat
        col_scale = r

    FlatOut = plot3Dparams.Cyl["Flat_or_Surf"] == 1  # If Flat, this is ==1

    # >>>> Plot using SURFACE function <<<<
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x, y, z * (1 - FlatOut), facecolors=plt.cm.viridis(col_scale), shade=False
    )

    ax.set_xlabel(r"$\theta \cos(\phi)$")
    ax.set_ylabel(r"$\theta \sin(\phi)$")
    ax.set_zlabel("Z")

    # Set plot-viewing angle
    if plot3Dparams.Cyl["Flat_or_Surf"] == 2:
        ax.view_init(elev=plot3Dparams.View[0], azim=plot3Dparams.View[1])
    else:
        ax.view_init(elev=90, azim=0)

    if plot3Dparams.cbo == 1:
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        mappable.set_array(col_scale)
        cbar = plt.colorbar(mappable, ax=ax)
        if (PT - 3) == 3:  # dB-plot
            yTick = cbar.get_ticks()
            yTick_dBn = -(1 - yTick) * plot3Dparams.dB_Lim
            cbar.set_ticks(yTick)
            cbar.set_ticklabels([f"{yt:.1f}" for yt in yTick_dBn])

    plt.show()


def plot_3D_Spherical(t, p, r, PT):
    # Check for dB and scale it in [0,1], so that it's plottable:
    if PT == 3:
        ro = r.copy()  # store originally-supplied r (needed for color-scaling)
        R = plot3Dparams.dB_Lim
        r = (np.maximum(r, -R) + R) / R

    # Convert from spherical (t,p,r) to cartesian (x,y,z) coordinates
    x = r * np.sin(np.radians(t)) * np.cos(np.radians(p))
    y = r * np.sin(np.radians(t)) * np.sin(np.radians(p))
    z = r * np.cos(np.radians(t))

    # If color-scaling was requested
    if plot3Dparams.ScaleColors == 1:
        if PT == 3:  # dB
            col_scale = 10.0 ** (ro / 10)
        elif PT == 1:  # |E| (linear)
            col_scale = r**2
        else:  # default P=U=|E|^2 (linear)
            col_scale = r
    else:
        col_scale = r

    # >>>> Plot using SURFACE function <<<<
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(col_scale), shade=False)

    # Set view
    ax.view_init(elev=plot3Dparams.View[0], azim=plot3Dparams.View[1])

    # If colorbar is on
    if plot3Dparams.cbo == 1:
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
        mappable.set_array(col_scale)
        cbar = plt.colorbar(mappable, ax=ax)
        if PT == 3:  # dB-plot
            yTick = cbar.get_ticks()
            yTick_dBn = -(1 - yTick) * plot3Dparams.dB_Lim
            cbar.set_ticks(yTick)
            cbar.set_ticklabels([f"{yt:.1f}" for yt in yTick_dBn])

    plt.show()


def plot_3D_Pattern(theta_deg, phi_deg, U_Lin, PlotType=None):
    global plot3Dparams

    if PlotType is None:
        PlotType = [1, 2, 3, 4, 5, 6]

    # Check theta and phi
    if theta_deg.ndim == 1 or phi_deg.ndim == 1:
        raise ValueError("## (theta,phi) must be meshgrid-produced 2D arrays")
    if np.max(theta_deg) < 90 or np.max(phi_deg) < 90:
        raise ValueError("## (theta,phi) must be in DEGREES")

    # Check U
    if np.any(U_Lin < 0):
        raise ValueError("## Make sure U is in linear power-scale")
    if not np.isreal(U_Lin).all():
        raise ValueError("## Make sure U is real")
    if U_Lin.shape != theta_deg.shape:
        raise ValueError("## Make sure U is same size as theta_deg and phi_deg")
    if np.max(U_Lin) != 1:
        print("** Normalizing U_Lin for max==1 (or 0dB)")
        U_Lin = U_Lin / np.max(U_Lin)

    # Plot
    for kp in range(len(PlotType)):
        plt.figure()
        # Prepare data, assuming U_Lin is intensity (units of power), Linear
        if PlotType[kp] % 3 == 1:  # PlotType 1, 4
            r = np.sqrt(U_Lin)
            titstr = "Ampl. |E| (Lin)"
        elif PlotType[kp] % 3 == 2:  # PlotType 2, 5
            r = U_Lin
            titstr = "Power |E|^2 (Lin.)"
        elif PlotType[kp] % 3 == 0:  # PlotType 3, 6
            r = 10 * np.log10(U_Lin)
            titstr = "Log (dB)"

        # Call appropriate plot function, for either 3D-spherical or 3D-cylindrical style
        if PlotType[kp] <= 3:
            plot_3D_Spherical(theta_deg, phi_deg, r, PlotType[kp])
            plt.title(f"{titstr} | Spher.")
        else:
            plot_3D_Cylindrical(theta_deg, phi_deg, r, PlotType[kp])
            plt.title(f"{titstr} | Cyl.")


# Example test case
if __name__ == "__main__":
    phi = np.arange(0, 361, 1)  # [deg]
    the = np.arange(0, 91, 1)  # [deg]
    phi_deg, theta_deg = np.meshgrid(phi, the)  # [deg]

    # Test plot for a 2D array of isotropic scatterers
    t_max = 30
    p_max = -120

    # Uniform 2D-array parameters
    Nx = 5  # [.] Number of elements of grid in x-dimension
    Ny = Nx  # [.] Number of elements of grid in y-dimension
    kdx = 2 * np.pi * 0.5  # [rad] kappa*dx, where dx: distance in x-dim
    kdy = kdx  # [rad] kappa*dy, where dy: distance in y-dim
    bx = (
        -kdx * np.sin(np.radians(t_max)) * np.cos(np.radians(p_max))
    )  # [rad] Phase difference in x-dim
    by = (
        -kdy * np.sin(np.radians(t_max)) * np.sin(np.radians(p_max))
    )  # [rad] Phase difference in y-dim

    # Relative phase-difference
    psix = (
        kdx * np.sin(np.radians(theta_deg)) * np.cos(np.radians(phi_deg)) + bx
    )  # [rad] x-dimension
    psiy = (
        kdy * np.sin(np.radians(theta_deg)) * np.sin(np.radians(phi_deg)) + by
    )  # [rad] y-dimension

    # Array Factor in 1D (x- and y-dimensions)
    with np.errstate(divide="ignore", invalid="ignore"):
        AF1Dx = np.sin(0.5 * Nx * psix) / np.sin(0.5 * psix)  # [.]
        AF1Dy = np.sin(0.5 * Ny * psiy) / np.sin(0.5 * psiy)  # [.]

    # Correct cases where division 0/0 --> "NaN" ("Not A Number") error
    AF1Dx[np.isnan(AF1Dx)] = (
        Nx
        * np.cos(0.5 * Nx * psix[np.isnan(AF1Dx)])
        / np.cos(0.5 * psix[np.isnan(AF1Dx)])
    )
    AF1Dy[np.isnan(AF1Dy)] = (
        Ny
        * np.cos(0.5 * Ny * psiy[np.isnan(AF1Dy)])
        / np.cos(0.5 * psiy[np.isnan(AF1Dy)])
    )

    # Array Factor in 2D (power)
    AF2D_power = np.abs(np.abs(AF1Dx) * np.abs(AF1Dy)) ** 2

    # Normalize
    AF2D_power_n = AF2D_power / np.max(AF2D_power)

    # Calculate Directivity
    Un = AF2D_power_n
    D = (
        4
        * np.pi
        * Un
        / np.trapezoid(np.trapezoid(Un * np.sin(np.radians(theta_deg)), axis=0), axis=0)
        / (np.pi / 180) ** 2
    )
    D0max = np.max(D)
    print(f"AF max-directivity is {10 * np.log10(D0max):.2f} [dBi]")

    U_Lin = D

    PlotType = [1, 2, 3, 4, 5, 6]
    plot_3D_Pattern(theta_deg, phi_deg, U_Lin, PlotType)
