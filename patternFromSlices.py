from dataclasses import dataclass

import numpy as np


@dataclass
class PatternPlotOptions:
    """Class for pattern plotting options"""

    transparency: float = 0.5
    magnitude_scale: str = "linear"
    size_ratio: float = 1.0
    antenna_offset: list[float] = None

    def __post_init__(self):
        if self.antenna_offset is None:
            self.antenna_offset = [0.0, 0.0, 0.0]


def pattern_from_slices(
    vert_slice: np.ndarray,
    theta: np.ndarray,
    horiz_slice: float | np.ndarray | None = None,
    phi: np.ndarray | None = None,
    method: str = "Summing",
    cross_weighted_normalization: float = 2.0,
    pattern_options: PatternPlotOptions | None = None,
    tol_nearest_angle_from_bs: float = 10.0,
    tol_gain_max_vs_bs: float = 3.0,
    tol_gain_diff_at_slice_intersect: list[float] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    # Input validation
    if not isinstance(vert_slice, np.ndarray) or not isinstance(theta, np.ndarray):
        raise ValueError("vert_slice and theta must be numpy arrays")

    if vert_slice.size != theta.size:
        raise ValueError("Dimension mismatch between vert_slice and theta")

    # Set defaults
    if horiz_slice is None:
        horiz_slice = np.max(vert_slice)
    if phi is None:
        phi = np.arange(0, 361, 5)
    if tol_gain_diff_at_slice_intersect is None:
        tol_gain_diff_at_slice_intersect = [1.0, 3.0]
    if pattern_options is None:
        pattern_options = PatternPlotOptions()

    # Convert scalar horiz_slice to array if needed
    if isinstance(horiz_slice, (int, float)):
        horiz_slice = np.full_like(phi, horiz_slice)
    elif horiz_slice.size != phi.size:
        raise ValueError("Dimension mismatch between horiz_slice and phi")

    # Check for repeated points
    check_repeated_points(vert_slice, theta, "el")
    check_repeated_points(horiz_slice, phi, "az")

    # Verify reconstruction requirements
    check_reconstruction_requirements(
        vert_slice,
        theta,
        horiz_slice,
        phi,
        tol_nearest_angle_from_bs,
        tol_gain_max_vs_bs,
        tol_gain_diff_at_slice_intersect,
    )

    # Normalize the data
    max_directivity = max(np.max(vert_slice), np.max(horiz_slice))
    vert_slice_norm = vert_slice - max_directivity
    horiz_slice_norm = horiz_slice - max_directivity

    # Preprocess data
    vert_mesh_log, theta_out, horiz_mesh_log, phi_out, back_plane, front_plane = (
        preprocess_data(vert_slice_norm, theta, horiz_slice_norm, phi, method)
    )

    # Calculate reconstructed 3D pattern
    if method.lower() == "summing":
        pat3d = vert_mesh_log + horiz_mesh_log
    elif method.lower() == "crossweighted":
        k = cross_weighted_normalization
        vert_mesh_lin = 10 ** (vert_mesh_log / 10)
        horiz_mesh_lin = 10 ** (horiz_mesh_log / 10)
        w1 = vert_mesh_lin * (1 - horiz_mesh_lin)
        w2 = horiz_mesh_lin * (1 - vert_mesh_lin)
        pat3d = (horiz_mesh_log * w1 + vert_mesh_log * w2) / np.power(
            w1**k + w2**k, 1 / k
        )
        pat3d[w1 == 0 & w2 == 0] = 0
    else:
        raise ValueError("Invalid method. Use 'Summing' or 'CrossWeighted'")

    # Denormalize the result
    pat3d = pat3d + max_directivity

    return pat3d, theta_out, phi_out


def check_repeated_points(vals: np.ndarray, angles: np.ndarray, az_or_el: str) -> None:
    """Check if directivity/gain values for repeated angles are equal"""
    angs = np.mod(angles, 360)

    # Find unique angles within tolerance
    unique_angs, inverse_indices = np.unique(angs, return_inverse=True)

    # Check for unequal values at repeated angles
    for idx in range(len(unique_angs)):
        mask = inverse_indices == idx
        if mask.sum() > 1:  # repeated angle
            if not np.allclose(vals[mask], vals[mask][0]):
                if az_or_el == "az":
                    raise ValueError(
                        f"Unequal values at repeated azimuth angle {unique_angs[idx]}"
                    )
                else:
                    raise ValueError(
                        f"Unequal values at repeated elevation angle {unique_angs[idx]}"
                    )


def wrap_to_alpha(x: float | np.ndarray, alpha: float) -> float | np.ndarray:
    """
    Helper function to wrap angles to the range (alpha-360, alpha].

    Args:
        x (float | np.ndarray): Angle(s) to wrap
        alpha (float): Reference angle

    Returns:
        float | np.ndarray: Wrapped angle(s)
    """
    return x - 360 * np.ceil((x - alpha) / 360)

def find_angle_closest_to_bs(angles: np.ndarray, angle_bs: float) -> tuple[float, int]:
    """
    Find the angle with minimum deviation from provided boresight angle.

    Args:
        angles (np.ndarray): Array of angles to search through
        angle_bs (float): Boresight angle reference

    Returns:
        tuple[float, int]: A tuple containing:
            - minimum absolute deviation from boresight angle
            - index of the angle with minimum deviation
    """
    # Center the interval (alpha-360, alpha] around angle_bs
    wrapped_angles = wrap_to_alpha(angles, angle_bs + 180)

    # Find minimum deviation and its index
    abs_deviations = np.abs(wrapped_angles - angle_bs)
    min_deviation = np.min(abs_deviations)
    min_idx = np.argmin(abs_deviations)

    return min_deviation, min_idx

# def find_angle_closest_to_bs(angles: np.ndarray, angle_bs: float) -> tuple[float, int]:
#     """Find angle with minimum deviation from boresight angle"""
#     # Simplify by directly calculating deviation from boresight
#     deviations = np.abs(angles - angle_bs)
#     idx = np.argmin(deviations)
#     min_deviation = deviations[idx]
#     return min_deviation, idx


def check_reconstruction_requirements(
    vert_slice: np.ndarray,
    theta: np.ndarray,
    horiz_slice: np.ndarray,
    phi: np.ndarray,
    tol_nearest_angle_from_bs: float,
    tol_gain_max_vs_bs: float,
    tol_gain_diff_at_slice_intersect: list[float],
) -> None:
    """Verify input data conformity to reconstruction algorithm requirements"""
    phi_bs = 0
    theta_bs = 90

    min_abs_theta_from_bs, idx_min_abs_theta = find_angle_closest_to_bs(theta, theta_bs)
    min_abs_phi_from_bs, idx_min_abs_phi = find_angle_closest_to_bs(phi, phi_bs)

    if min_abs_theta_from_bs > tol_nearest_angle_from_bs:
        raise ValueError(
            f"No angles near boresight in vertical slice (tolerance: {tol_nearest_angle_from_bs}°)"
        )

    if min_abs_phi_from_bs > tol_nearest_angle_from_bs:
        raise ValueError(
            f"No angles near boresight in horizontal slice (tolerance: {tol_nearest_angle_from_bs}°)"
        )

    if (np.max(vert_slice) - vert_slice[idx_min_abs_theta]) > tol_gain_max_vs_bs:
        raise ValueError(
            f"Gain at boresight vs max gain in vertical slice exceeds tolerance ({tol_gain_max_vs_bs} dB)"
        )

    if (np.max(horiz_slice) - horiz_slice[idx_min_abs_phi]) > tol_gain_max_vs_bs:
        raise ValueError(
            f"Gain at boresight vs max gain in horizontal slice exceeds tolerance ({tol_gain_max_vs_bs} dB)"
        )

    gain_diff = abs(vert_slice[idx_min_abs_theta] - horiz_slice[idx_min_abs_phi])
    if gain_diff > tol_gain_diff_at_slice_intersect[1]:
        raise ValueError(
            f"Gain difference at slice intersection exceeds tolerance ({tol_gain_diff_at_slice_intersect[1]} dB)"
        )
    elif gain_diff > tol_gain_diff_at_slice_intersect[0]:
        print(
            f"Warning: Gain difference at slice intersection exceeds warning threshold ({tol_gain_diff_at_slice_intersect[0]} dB)"
        )


def preprocess_data(
    vert_slice_norm: np.ndarray,
    theta: np.ndarray,
    horiz_slice_norm: np.ndarray,
    phi: np.ndarray,
    reconstruct_method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict, dict]:
    """Preprocess data for reconstruction"""
    theta_mod_360 = np.mod(theta, 360)
    phi_mod_360 = np.mod(phi, 360)

    if reconstruct_method.lower() in ["summing", "crossweighted"]:
        idx_phi = np.arange(len(phi))
        idx_theta = theta_mod_360 <= 180
    else:
        raise ValueError("Invalid reconstruction method")

    idx_p = (phi_mod_360 <= 90) | (phi_mod_360 >= 270)
    idx_t = (theta_mod_360 <= 180) | (theta_mod_360 >= 0)

    back_plane = {"P": np.where(~idx_p)[0], "T": np.where(~idx_t)[0]}
    front_plane = {"P": np.where(idx_p)[0], "T": np.where(idx_t)[0]}

    theta_out = theta[idx_theta]
    phi_out = phi[idx_phi]

    # Create meshgrid
    phi_mesh, theta_mesh = np.meshgrid(
        vert_slice_norm[idx_theta], horiz_slice_norm[idx_phi]
    )

    if len(theta_out) != len(theta):
        print("Warning: Back plane data has been discarded")

    return phi_mesh, theta_out, theta_mesh, phi_out, back_plane, front_plane
