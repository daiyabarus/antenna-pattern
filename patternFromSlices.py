import warnings

import numpy as np


def pattern_from_slices(
    vert_slice,
    theta,
    horiz_slice=None,
    phi=None,
    method="Summing",
    cross_weighted_normalization=2,
    pattern_options=None,
    tol_nearest_angle_from_boresight=10,
    tol_gain_max_vs_boresight=3,
    tol_gain_diff_at_slice_intersect=(1, 3),
):
    """
    Reconstructs an approximate 3D radiation pattern from two orthogonal pattern slices.

    vert_slice: 2D pattern slice data (in dBI) along the vertical/elevation plane (list or np.array)
    theta: Polar/inclination angles in degrees in spherical coordinates (list or np.array)

    horiz_slice: Optional. 2D pattern slice data (in dBI) along the horizontal/azimuthal plane.
                 Can be a scalar for omnidirectionality (azimuthal symmetry), or a vector.
    phi: Optional. Azimuthal angles in degrees in the spherical coordinate system (list or np.array).
    method: Optional. Algorithm used for the reconstruction ('Summing' or 'CrossWeighted'). Default is 'Summing'.
    cross_weighted_normalization: Optional. Normalization parameter for Cross-Weighted summing method. Default is 2.
    pattern_options: Optional. Object to specify pattern plot properties like Transparency, MagnitudeScale. Not implemented in this version.

    Returns:
    pat3D: Reconstructed 3D pattern as a matrix
    theta_out: Corresponding theta values (degrees)
    phi_out: Corresponding phi values (degrees)
    """
    # Check input dimensions
    if len(vert_slice) != len(theta):
        raise ValueError("Dimensions of vert_slice and theta do not match.")

    if phi is None:
        phi = np.arange(0, 361, 5)

    if horiz_slice is None:
        horiz_slice = np.max(vert_slice) * np.ones_like(phi)
    elif np.isscalar(horiz_slice):
        horiz_slice = horiz_slice * np.ones_like(phi)
    elif len(horiz_slice) != len(phi):
        raise ValueError("Dimensions of horiz_slice and phi do not match.")

    # Check for repeated points in slices
    check_repeated_points(vert_slice, theta, "el")
    check_repeated_points(horiz_slice, phi, "az")

    # Verify reconstruction requirements
    check_reconstruction_requirements(
        vert_slice,
        theta,
        horiz_slice,
        phi,
        tol_nearest_angle_from_boresight,
        tol_gain_max_vs_boresight,
        tol_gain_diff_at_slice_intersect,
    )

    # Normalize the data
    max_directivity = max(np.max(vert_slice), np.max(horiz_slice))
    vert_slice_norm = vert_slice - max_directivity
    horiz_slice_norm = horiz_slice - max_directivity

    # Reconstruction method
    if method == "Summing":
        vert_mesh_log, theta_out, horiz_mesh_log, phi_out = preprocess_data(
            vert_slice_norm, theta, horiz_slice_norm, phi, method
        )
        pat3D = vert_mesh_log + horiz_mesh_log
    elif method == "CrossWeighted":
        vert_mesh_log, theta_out, horiz_mesh_log, phi_out = preprocess_data(
            vert_slice_norm, theta, horiz_slice_norm, phi, method
        )
        k = cross_weighted_normalization
        vert_mesh_lin = 10 ** (vert_mesh_log / 10)
        horiz_mesh_lin = 10 ** (horiz_mesh_log / 10)
        w1 = vert_mesh_lin * (1 - horiz_mesh_lin)
        w2 = horiz_mesh_lin * (1 - vert_mesh_lin)
        pat3D = (horiz_mesh_log * w1 + vert_mesh_log * w2) / np.cbrt(w1**k + w2**k)
        pat3D[np.logical_and(w1 == 0, w2 == 0)] = 0
    else:
        raise ValueError(f"Unknown method: {method}")

    # Denormalize the result
    pat3D = pat3D + max_directivity

    return pat3D, theta_out, phi_out


def check_repeated_points(vals, angles, az_or_el):
    """
    Ensures that directivity/gain values for each set of repeated angles are equal.
    """
    unique_angles, indices = np.unique(np.round(angles), return_inverse=True)
    repeated_vals = np.array(
        [len(np.unique(vals[indices == i])) > 1 for i in range(len(unique_angles))]
    )

    if any(repeated_vals):
        repeated_angles = unique_angles[repeated_vals]
        raise ValueError(
            f"Repeated angles with unequal values in {az_or_el}: {repeated_angles[0]}"
        )


def check_reconstruction_requirements(
    vert_slice,
    theta,
    horiz_slice,
    phi,
    tol_nearest_angle_from_boresight,
    tol_gain_max_vs_boresight,
    tol_gain_diff_at_slice_intersect,
):
    """
    Ensures that the input data conforms to the reconstruction algorithm's requirements.
    """
    phi_bs = 0
    theta_bs = 90
    min_theta_from_bs = np.min(np.abs(np.mod(theta - theta_bs, 360)))
    min_phi_from_bs = np.min(np.abs(np.mod(phi - phi_bs, 360)))

    if min_theta_from_bs > tol_nearest_angle_from_boresight:
        raise ValueError(
            f"No angles near boresight in vertical slice (tolerance: {tol_nearest_angle_from_boresight})"
        )

    if min_phi_from_bs > tol_nearest_angle_from_boresight:
        raise ValueError(
            f"No angles near boresight in horizontal slice (tolerance: {tol_nearest_angle_from_boresight})"
        )

    if (
        np.max(vert_slice) - vert_slice[np.argmin(np.abs(theta - theta_bs))]
    ) > tol_gain_max_vs_boresight:
        raise ValueError("Vertical slice gain at boresight exceeds tolerance.")

    if (
        np.max(horiz_slice) - horiz_slice[np.argmin(np.abs(phi - phi_bs))]
    ) > tol_gain_max_vs_boresight:
        raise ValueError("Horizontal slice gain at boresight exceeds tolerance.")

    # Check gain difference at slice intersection
    if (
        np.abs(
            vert_slice[np.argmin(np.abs(theta - theta_bs))]
            - horiz_slice[np.argmin(np.abs(phi - phi_bs))]
        )
        > tol_gain_diff_at_slice_intersect[1]
    ):
        raise ValueError("Gain difference at slice intersection exceeds tolerance.")
    elif (
        np.abs(
            vert_slice[np.argmin(np.abs(theta - theta_bs))]
            - horiz_slice[np.argmin(np.abs(phi - phi_bs))]
        )
        > tol_gain_diff_at_slice_intersect[0]
    ):
        warnings.warn("Gain difference at slice intersection is significant.")


def preprocess_data(vert_slice_norm, theta, horiz_slice_norm, phi, method):
    """
    Preprocess data according to the chosen reconstruction method.
    """
    theta_mod360 = np.mod(theta, 360)
    phi_mod360 = np.mod(phi, 360)

    if method in ["Summing", "CrossWeighted"]:
        idx_theta = theta_mod360 <= 180
        idx_phi = np.arange(len(phi))  # Use all phi values

    theta_out = theta[idx_theta]
    phi_out = phi[idx_phi]
    vert_mesh_log, horiz_mesh_log = np.meshgrid(
        vert_slice_norm[idx_theta], horiz_slice_norm[idx_phi]
    )

    return vert_mesh_log, theta_out, horiz_mesh_log, phi_out
