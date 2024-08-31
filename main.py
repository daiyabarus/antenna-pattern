import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pyvista as pv
import streamlit as st
from stpyvista import stpyvista

from styles import styling


@dataclass
class AntennaPattern:
    horizontal: pd.DataFrame
    vertical: pd.DataFrame
    gain: float
    tilt: float
    beamwidth: float
    h_beamwidth: float
    v_beamwidth: float
    v_hor_cut: float
    h_ver_cut: float
    tx_power: float
    frequency: float
    wavelength: float = 0.0

    def __post_init__(self):
        self.wavelength = 3e8 / (self.frequency * 1e6)


def watts_to_dbm(watts):
    return 10 * np.log10(watts * 1000)


def dbm_to_watts(dbm):
    return 10 ** ((dbm - 30) / 10)


def decode_file_content(file_path: str) -> pd.DataFrame | None:
    """Decode the file content from the provided file path with multiple encodings."""
    encodings = ["utf-8", "iso-8859-1", "windows-1252"]

    def try_decode(encoding):
        try:
            with open(file_path, encoding=encoding) as file:
                content = file.read()
            lines = content.split("\n")
            headers = lines[0].strip().split("\t")
            return pd.DataFrame(
                [
                    dict(zip(headers, line.strip().split("\t")))
                    for line in lines[1:]
                    if line.strip()
                ]
            )
        except UnicodeDecodeError:
            return None

    return next((df for df in map(try_decode, encodings) if df is not None), None)


def extract_pattern(pattern_text: str) -> tuple[str | None, str | None]:
    """Extract horizontal and vertical patterns from the pattern text."""
    matches = re.findall(r"(?<=360\s)(.*?\s+359\s+\d+\.\d+)", pattern_text)
    return (matches[0], matches[1]) if len(matches) >= 2 else (None, None)


def extract_tx_power(comment: str) -> float:
    """Extract transmission power from the comment string."""
    power_match = re.search(r"Tx Power=(\d+)x(\d+)\s*dBm", comment)
    if power_match:
        num_channels = int(power_match.group(1))
        power_per_channel = int(power_match.group(2))
        total_power_dbm = 10 * np.log10(num_channels * (10 ** (power_per_channel / 10)))
        return total_power_dbm
    return 0


def transpose_pattern(pattern_text: str) -> pd.DataFrame:
    """Transpose the pattern text into a pandas DataFrame."""
    values = pattern_text.strip().split()
    return pd.DataFrame(
        {
            "angle": [float(angle) for angle in values[::2]],
            "Att dB": [float(att) for att in values[1::2]],
        }
    )


def extract_antenna_parameters(comment: str) -> dict[str, float]:
    """Extract antenna parameters from the comment string."""
    params = {
        "h_beamwidth": 0,
        "v_beamwidth": 0,
        "v_hor_cut": 0,
        "h_ver_cut": 0,
        "tx_power": 40,
    }

    beam_match = re.search(r"H-(\d+)_V-(\d+)", comment)
    if beam_match:
        params["h_beamwidth"] = float(beam_match.group(1))
        params["v_beamwidth"] = float(beam_match.group(2))

    cut_match = re.search(r"V_HorCut=(\d+),\s*H_VerCut=(\d+)", comment)
    if cut_match:
        params["v_hor_cut"] = float(cut_match.group(1))
        params["h_ver_cut"] = float(cut_match.group(2))

    power_match = re.search(r"Tx Power=(\d+)x(\d+)\s*dBm", comment)
    if power_match:
        num_channels = int(power_match.group(1))
        power_per_channel = int(power_match.group(2))
        total_power_watts = num_channels * (10 ** (power_per_channel / 10))
        params["tx_power"] = 10 * np.log10(total_power_watts)

    return params


@st.cache_resource
def create_3d_chart(
    pattern: AntennaPattern, power_adjustment: float, default_power: float
):
    """Create a 3D antenna pattern chart using PyVista."""
    theta = np.radians(np.linspace(0, 360, 361))
    phi = np.radians(np.linspace(0, 180, 181))
    THETA, PHI = np.meshgrid(theta, phi)

    h_angles = np.radians(pattern.horizontal["angle"].values)
    h_gain = pattern.horizontal["Att dB"].values
    v_angles = np.radians(pattern.vertical["angle"].values)
    v_gain = pattern.vertical["Att dB"].values

    h_gain_interp = np.interp(theta, h_angles, h_gain, period=2 * np.pi)
    v_gain_interp = np.interp(phi, v_angles, v_gain)

    R = np.outer(v_gain_interp, h_gain_interp)
    R_adjusted = R - (power_adjustment - default_power)
    max_gain = pattern.gain + (power_adjustment - default_power)

    X = (max_gain - R_adjusted) * np.sin(PHI) * np.cos(THETA)
    Y = (max_gain - R_adjusted) * np.sin(PHI) * np.sin(THETA)
    Z = (max_gain - R_adjusted) * np.cos(PHI)

    grid = pv.StructuredGrid(X, Y, Z)
    attenuation = (max_gain - R_adjusted).flatten(order="F")
    grid["attenuation"] = attenuation

    plotter = pv.Plotter()
    plotter.add_mesh(
        grid,
        scalars="attenuation",
        cmap=["red", "yellow", "green", "blue"],
        show_scalar_bar=True,
        smooth_shading=True,
        show_edges=False,
    )

    plotter.background_color = "white"
    plotter.view_isometric()
    return plotter


def create_polar_chart(df: pd.DataFrame, color: str, max_gain: float) -> go.Figure:
    """Create a polar chart using Plotly for the antenna pattern."""
    gain = -df["Att dB"]
    dBi_values = max_gain + gain

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=gain,
            theta=df["angle"],
            mode="lines",
            name="Antenna Pattern",
            line=dict(color=color, width=2),
            hovertemplate="Angle: %{theta:.2f}°<br>Attenuation: %{r:.2f} dB<br>Gain: %{text:.2f} dBi<extra></extra>",
            text=dBi_values,
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[gain.min() - 2, 0],
                tickmode="array",
                tickvals=np.arange(gain.min() - 2, 2, 2),
                ticktext=[
                    str(abs(int(val))) for val in np.arange(gain.min() - 2, 2, 2)
                ],
            ),
            angularaxis=dict(
                tickmode="array",
                tickvals=[0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330],
                direction="clockwise",
            ),
        ),
        showlegend=False,
        height=500,
        width=500,
    )

    return fig


def create_3d_chart_plotly(
    patterns: AntennaPattern | list[AntennaPattern],
    power_adjustment: float,
    default_power: float,
    array_config: dict = {"rows": 1, "columns": 1, "spacing": 0.5},
) -> go.Figure:
    """
    Create a 3D chart of the antenna pattern using Plotly.

    :param patterns: AntennaPattern object or list of AntennaPattern objects
    :param power_adjustment: Power adjustment in dBm
    :param default_power: Default power in dBm
    :param array_config: Dictionary containing array configuration
    :return: Plotly Figure object
    """
    # Ensure patterns is a list
    if isinstance(patterns, AntennaPattern):
        patterns = [patterns]

    # Define theta and phi as radians
    theta = np.radians(np.linspace(0, 360, 361))
    phi = np.radians(np.linspace(0, 180, 181))
    THETA, PHI = np.meshgrid(theta, phi)

    # Initialize array factor
    array_factor = np.zeros_like(THETA, dtype=complex)

    # Calculate array factor
    for i, pattern in enumerate(patterns):
        # Calculate element position
        row = i // array_config["columns"]
        col = i % array_config["columns"]
        x_pos = (col - (array_config["columns"] - 1) / 2) * array_config["spacing"]
        y_pos = (row - (array_config["rows"] - 1) / 2) * array_config["spacing"]

        # Interpolate gains
        h_angles = np.radians(pattern.horizontal["angle"].values)
        h_gain = -pattern.horizontal["Att dB"].values
        v_angles = np.radians(pattern.vertical["angle"].values)
        v_gain = -pattern.vertical["Att dB"].values

        h_gain_interp = np.interp(theta, h_angles, h_gain, period=2 * np.pi)
        v_gain_interp = np.interp(phi, v_angles, v_gain)

        # Combine gains and convert to linear scale
        element_gain = np.outer(v_gain_interp, h_gain_interp)
        element_gain_linear = 10 ** (element_gain / 20)

        # Calculate phase shift due to element position
        phase_shift = (
            2
            * np.pi
            * (
                x_pos * np.sin(PHI) * np.cos(THETA)
                + y_pos * np.sin(PHI) * np.sin(THETA)
            )
        )

        # Add element contribution to array factor
        array_factor += element_gain_linear * np.exp(1j * phase_shift)

    # Calculate total pattern
    total_pattern = 20 * np.log10(np.abs(array_factor))

    # Adjust for power
    total_pattern_adjusted = total_pattern + power_adjustment - default_power

    # Convert to Cartesian coordinates
    X = total_pattern_adjusted * np.sin(PHI) * np.cos(THETA)
    Y = total_pattern_adjusted * np.sin(PHI) * np.sin(THETA)
    Z = total_pattern_adjusted * np.cos(PHI)

    # Create the 3D surface plot
    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=total_pattern_adjusted,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Gain (dBi)"),
            )
        ]
    )

    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(showgrid=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, showticklabels=False, title=""),
        ),
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0),
        title="3D Antenna Radiation Pattern",
    )

    return fig


def create_3d_chart_polar(
    df_horizontal: pd.DataFrame, df_vertical: pd.DataFrame, color: str, max_gain: float
) -> go.Figure:
    """Create a 3D antenna pattern chart from horizontal and vertical polar data using Plotly."""
    # Convert angles to radians for spherical to Cartesian transformation
    theta = np.radians(df_horizontal["angle"].values)  # Horizontal angles (azimuth)
    phi = np.radians(df_vertical["angle"].values)  # Vertical angles (elevation)

    # Calculate the gain (in dB) from the attenuation
    h_gain = max_gain - df_horizontal["Att dB"].values
    v_gain = max_gain - df_vertical["Att dB"].values

    # Interpolating the gains for each angle pair
    THETA, PHI = np.meshgrid(theta, phi)
    R = np.outer(v_gain, h_gain)  # Combine gains

    # Spherical to Cartesian transformation
    X = R * np.sin(PHI) * np.cos(THETA)
    Y = R * np.sin(PHI) * np.sin(THETA)
    Z = R * np.cos(PHI)

    # Create the 3D surface plot using Plotly
    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=R,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Gain (dBi)"),
            )
        ]
    )

    # Update layout for better visualization
    fig.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(showgrid=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, showticklabels=False, title=""),
            zaxis=dict(showgrid=False, showticklabels=False, title=""),
        ),
        title="3D Antenna Radiation Pattern",
        width=800,
        height=800,
        margin=dict(l=0, r=0, b=0, t=0),
    )

    return fig


def main():
    # Define the path to the fixed file
    script_dir = os.path.dirname(__file__)
    fixed_file_path = os.path.join(script_dir, "AIR6488.txt")
    # in wavelengths

    # Load and cache the file content
    if "df" not in st.session_state:
        st.session_state.df = decode_file_content(fixed_file_path)

    if st.session_state.df is not None:
        if (
            "Comments" in st.session_state.df.columns
            and "Max Frequency (MHz)" in st.session_state.df.columns
        ):
            comments = st.session_state.df["Comments"].unique()
            selected_comment = st.selectbox(
                "Select Ant Type", comments, key="selected_comment"
            )

            selected_row = st.session_state.df[
                st.session_state.df["Comments"] == selected_comment
            ].iloc[0]
            horizontal_pattern, vertical_pattern = extract_pattern(
                selected_row["Pattern"]
            )

            if all([horizontal_pattern, vertical_pattern]):
                antenna_params = extract_antenna_parameters(selected_row["Comments"])

                pattern = AntennaPattern(
                    horizontal=transpose_pattern(horizontal_pattern),
                    vertical=transpose_pattern(vertical_pattern),
                    gain=float(selected_row["Gain (dBi)"]),
                    tilt=float(selected_row["Pattern Electrical Tilt (°)"]),
                    beamwidth=float(selected_row["Half-power Beamwidth"]),
                    frequency=float(selected_row["Max Frequency (MHz)"]),
                    **antenna_params,
                )

                # Set default power to 200 watts
                default_power_watts = 100.0
                default_power_dbm = watts_to_dbm(default_power_watts)

                power_adjustment_watts = st.number_input(
                    "Tx Power (Watts)",
                    value=float(default_power_watts),
                    min_value=0.1,
                    max_value=200.0,
                    step=0.1,
                    format="%.1f",
                    key="power_adjustment_watts",
                )

                power_adjustment_dbm = watts_to_dbm(power_adjustment_watts)

                if power_adjustment_dbm is not None:
                    power_diff = power_adjustment_dbm - default_power_dbm
                    adjusted_gain = float(selected_row["Gain (dBi)"]) + power_diff

                    st.markdown("#")
                    col1, col2, col3 = st.columns(3)
                    st.markdown("#")
                    with col1:
                        st.markdown(
                            *styling(
                                "Horizontal Pattern",
                                tag="h6",
                                font_size=18,
                                text_align="center",
                            )
                        )
                        fig_h = create_polar_chart(
                            pattern.horizontal, "red", adjusted_gain
                        )
                        st.plotly_chart(fig_h, use_container_width=True)

                    with col2:
                        st.markdown(
                            *styling(
                                "Vertical Pattern",
                                tag="h6",
                                font_size=18,
                                text_align="center",
                            )
                        )
                        fig_v = create_polar_chart(
                            pattern.vertical, "green", adjusted_gain
                        )
                        st.plotly_chart(fig_v, use_container_width=True)

                    with col3:
                        st.markdown(
                            *styling(
                                "Antenna Data",
                                tag="h6",
                                font_size=18,
                                text_align="center",
                            )
                        )
                        table_data = f"""
                        | **Parameter**                    | **Value**                                                        |
                        |:---------------------------------|:-----------------------------------------------------------------|
                        | **Name**                         | {selected_row['Name']}                                           |
                        | **Gain**                         | {selected_row['Gain (dBi)']} dBi                                 |
                        | **Tx Power**                     | {power_adjustment_watts:.2f} W ({power_adjustment_dbm:.2f} dBm)  |
                        | **Adjusted Gain**                | {adjusted_gain:.2f} dBi                                          |
                        | **Pattern Electrical Tilt**      | {selected_row['Pattern Electrical Tilt (°)']}°                   |
                        | **Half-power Beamwidth**         | {selected_row['Half-power Beamwidth']}°                          |
                        | **Frequency Range**              | {selected_row['Min Frequency (MHz)']} - {selected_row['Max Frequency (MHz)']} MHz |
                        | **Pattern Electrical Azimuth**   | {selected_row['Pattern Electrical Azimuth (°)']}°                |
                        | **Remark**                       | {selected_row['Comments']}                                       |
                        """
                        st.markdown(table_data)
                    array_config = {"rows": 4, "columns": 8, "spacing": 0.5}
                    st.markdown("#")
                    st.markdown(
                        *styling(
                            "🗼 3D Antenna Pattern",
                            tag="h5",
                            font_size=32,
                            text_align="center",
                        )
                    )
                    col1, col2 = st.columns(2)
                    with col1:
                        plotter = create_3d_chart(
                            pattern, power_adjustment_dbm, default_power_dbm
                        )
                        stpyvista(
                            plotter,
                            key=f"antenna_pattern_{selected_comment}_{default_power_dbm}",
                        )
                    with col2:
                        fig_3d = create_3d_chart_plotly(
                            pattern,
                            power_adjustment_dbm,
                            default_power_dbm,
                            array_config,
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
                        fig_3d = create_3d_chart_plotly(
                            pattern,
                            power_adjustment_dbm,
                            default_power_dbm,
                            array_config,
                        )
                        st.plotly_chart(fig_3d, use_container_width=True)
            else:
                st.error("Could not extract pattern data for the selected comment.")
        else:
            st.error(
                'The uploaded file does not contain the expected "Comments" or "Max Frequency (MHz)" column.'
            )


if __name__ == "__main__":
    st.set_page_config(page_title="Antenna Pattern", layout="wide", page_icon="🧊")
    main()
