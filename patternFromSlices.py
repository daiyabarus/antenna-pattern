import warnings

import numpy as np


def pattern_from_slices(
    vert_slice,
    theta,
    horiz_slice=None,
    phi=None,
    method="Summing",
    cross_weighted_normalization=2,
    tol_nearest_angle_from_boresight=10,
    tol_gain_max_vs_boresight=3,
    tol_gain_diff_at_slice_intersect=(1, 3),
):
    """
    Reconstructs an approximate 3D radiation pattern from two orthogonal pattern slices.
    """
    if len(vert_slice) != len(theta):
        raise ValueError("Dimensions of vert_slice and theta do not match.")

    phi = phi if phi is not None else np.arange(0, 361, 5)
    horiz_slice = (
        horiz_slice
        if horiz_slice is not None
        else (
            np.max(vert_slice) * np.ones_like(phi)
            if np.isscalar(horiz_slice)
            else horiz_slice
        )
    )

    if len(horiz_slice) != len(phi):
        raise ValueError("Dimensions of horiz_slice and phi do not match.")

    check_repeated_points(vert_slice, theta, "el")
    check_repeated_points(horiz_slice, phi, "az")
    check_reconstruction_requirements(
        vert_slice,
        theta,
        horiz_slice,
        phi,
        tol_nearest_angle_from_boresight,
        tol_gain_max_vs_boresight,
        tol_gain_diff_at_slice_intersect,
    )

    max_directivity = max(np.max(vert_slice), np.max(horiz_slice))
    vert_slice_norm = vert_slice - max_directivity
    horiz_slice_norm = horiz_slice - max_directivity

    vert_mesh_log, theta_out, horiz_mesh_log, phi_out = preprocess_data(
        vert_slice_norm, theta, horiz_slice_norm, phi, method
    )

    if method == "CrossWeighted":
        vert_mesh_lin = 10 ** (vert_mesh_log / 10)
        horiz_mesh_lin = 10 ** (horiz_mesh_log / 10)
        w1 = vert_mesh_lin * (1 - horiz_mesh_lin)
        w2 = horiz_mesh_lin * (1 - vert_mesh_lin)
        pat3D = (horiz_mesh_log * w1 + vert_mesh_log * w2) / np.cbrt(
            w1**cross_weighted_normalization + w2**cross_weighted_normalization
        )
        pat3D[np.logical_and(w1 == 0, w2 == 0)] = 0
    else:
        pat3D = vert_mesh_log + horiz_mesh_log

    return pat3D + max_directivity, theta_out, phi_out


def check_repeated_points(vals, angles, az_or_el):
    unique_angles, indices = np.unique(np.round(angles), return_inverse=True)
    repeated_vals = np.array(
        [len(np.unique(vals[indices == i])) > 1 for i in range(len(unique_angles))]
    )
    if any(repeated_vals):
        raise ValueError(
            f"Repeated angles with unequal values in {az_or_el}: {unique_angles[repeated_vals][0]}"
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
    phi_bs, theta_bs = 0, 90
    if np.min(np.abs(np.mod(theta - theta_bs, 360))) > tol_nearest_angle_from_boresight:
        raise ValueError("No angles near boresight in vertical slice.")
    if np.min(np.abs(np.mod(phi - phi_bs, 360))) > tol_nearest_angle_from_boresight:
        raise ValueError("No angles near boresight in horizontal slice.")
    if (
        np.max(vert_slice) - vert_slice[np.argmin(np.abs(theta - theta_bs))]
        > tol_gain_max_vs_boresight
    ):
        raise ValueError("Vertical slice gain at boresight exceeds tolerance.")
    if (
        np.max(horiz_slice) - horiz_slice[np.argmin(np.abs(phi - phi_bs))]
        > tol_gain_max_vs_boresight
    ):
        raise ValueError("Horizontal slice gain at boresight exceeds tolerance.")

    gain_diff = np.abs(
        vert_slice[np.argmin(np.abs(theta - theta_bs))]
        - horiz_slice[np.argmin(np.abs(phi - phi_bs))]
    )
    if gain_diff > tol_gain_diff_at_slice_intersect[1]:
        raise ValueError("Gain difference at slice intersection exceeds tolerance.")
    elif gain_diff > tol_gain_diff_at_slice_intersect[0]:
        warnings.warn("Gain difference at slice intersection is significant.")


def preprocess_data(vert_slice_norm, theta, horiz_slice_norm, phi, method):
    theta_mod360 = np.mod(theta, 360)
    idx_theta = theta_mod360 <= 180
    vert_mesh_log, horiz_mesh_log = np.meshgrid(
        vert_slice_norm[idx_theta], horiz_slice_norm
    )
    return vert_mesh_log, theta[idx_theta], horiz_mesh_log, phi
