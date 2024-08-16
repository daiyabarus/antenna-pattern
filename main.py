import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


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


def split_txt_file(uploaded_file):
    encodings = ["utf-8", "iso-8859-1", "windows-1252"]

    def try_decode(encoding):
        try:
            content = uploaded_file.getvalue().decode(encoding)
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

    df = next((df for df in map(try_decode, encodings) if df is not None), None)
    if df is None:
        st.error("Unable to decode the file. Please check the file encoding.")
    return df


def extract_pattern(pattern_text: str) -> tuple[str | None, str | None]:
    matches = re.findall(r"(?<=360\s)(.*?\s+359\s+\d+\.\d+)", pattern_text)
    return (matches[0], matches[1]) if len(matches) >= 2 else (None, None)


def extract_tx_power(comment: str) -> float:
    power_match = re.search(r"Tx Power=(\d+)x(\d+)\s*dBm", comment)
    if power_match:
        num_channels = int(power_match.group(1))
        power_per_channel = int(power_match.group(2))
        total_power_dbm = 10 * np.log10(num_channels * (10 ** (power_per_channel / 10)))
        return total_power_dbm
    return 0


def transpose_pattern(pattern_text: str) -> pd.DataFrame:
    values = pattern_text.strip().split()
    return pd.DataFrame(
        {
            "angle": [float(angle) for angle in values[::2]],
            "Att dB": [float(att) for att in values[1::2]],
        }
    )


def extract_antenna_parameters(comment: str) -> dict[str, float]:
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


def create_3d_chart(
    pattern: AntennaPattern, power_adjustment: float, default_power: float
) -> go.Figure:
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
    R_norm = (R_adjusted - R_adjusted.min()) / (R_adjusted.max() - R_adjusted.min())

    max_gain = pattern.gain + (power_adjustment - default_power)
    X = (max_gain - R_adjusted) * np.sin(PHI) * np.cos(THETA)
    Y = (max_gain - R_adjusted) * np.sin(PHI) * np.sin(THETA)
    Z = (max_gain - R_adjusted) * np.cos(PHI)

    # Calculate dBi values
    dBi_values = max_gain - R_adjusted

    fig = go.Figure(
        data=[
            go.Surface(
                x=X,
                y=Y,
                z=Z,
                surfacecolor=R_norm,
                colorscale=[
                    (0, "rgb(15, 245, 22)"),  # Green
                    (0.2, "rgb(126, 245, 15)"),  # Light green
                    (0.4, "rgb(247, 243, 15)"),  # Yellow
                    (0.6, "rgb(245, 162, 15)"),  # Orange
                    (0.8, "rgb(245, 84, 5)"),  # Light red
                    (1, "rgb(245, 5, 5)"),  # Red
                ],
                colorbar=dict(title="Attenuation (dB)"),
                hovertemplate="X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<br>Gain: %{surfacecolor:.2f} dBi<extra></extra>",
            )
        ]
    )

    fig.update_layout(
        title=f"3D Antenna Pattern (Power: {power_adjustment:.2f} dBm, Adjustment: {power_adjustment - default_power:.2f} dB)",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8), up=dict(x=0, y=0, z=1)),
        ),
        autosize=False,
        width=700,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
    )

    return fig


def create_polar_chart(
    df: pd.DataFrame, title: str, color: str, max_gain: float
) -> go.Figure:
    gain = -df["Att dB"]
    dBi_values = max_gain + gain  # Calculate dBi values

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=gain,
            theta=df["angle"],
            mode="lines",
            name="Antenna Pattern",
            line=dict(color=color, width=2),
            hovertemplate="Angle: %{theta:.2f}°<br>Attenuation: %{r:.2f} dB<br>Gain: %{text:.2f} dBi<extra></extra>",
            text=dBi_values,  # Add dBi values for hover information
        )
    )

    fig.update_layout(
        title=title,
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


def main():
    st.set_page_config(
        layout="wide",
    )
    uploaded_file = st.file_uploader("Choose Antenna Pattern file", type="txt")
    if uploaded_file is not None:
        df = split_txt_file(uploaded_file)

        if (
            df is not None
            and "Comments" in df.columns
            and "Max Frequency (MHz)" in df.columns
        ):
            comments = df["Comments"].unique()
            selected_comment = st.selectbox("Select Ant Type", comments)

            selected_row = df[df["Comments"] == selected_comment].iloc[0]
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

                default_power = extract_tx_power(selected_row["Comments"])
                power_adjustment = st.number_input(
                    "Tx Power (dBm)", value=default_power, step=0.1
                )

                adjusted_gain = float(selected_row["Gain (dBi)"]) + (
                    power_adjustment - default_power
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.subheader("Horizontal Pattern")
                    fig_h = create_polar_chart(
                        pattern.horizontal, "Horizontal Pattern", "red", adjusted_gain
                    )
                    st.plotly_chart(fig_h, use_container_width=True)

                with col2:
                    st.subheader("Vertical Pattern")
                    fig_v = create_polar_chart(
                        pattern.vertical, "Vertical Pattern", "green", adjusted_gain
                    )
                    st.plotly_chart(fig_v, use_container_width=True)

                st.subheader("3D Antenna Pattern")
                fig_3d = create_3d_chart(pattern, power_adjustment, default_power)
                st.plotly_chart(fig_3d, use_container_width=True)
                with col3:
                    st.subheader("3D Antenna Pattern")
                    fig_3d = create_3d_chart(pattern, power_adjustment, default_power)
                    st.plotly_chart(fig_3d, use_container_width=True)

                st.subheader("Antenna Characteristics")
                st.write(f"Name: {selected_row['Name']}")
                st.write(f"Gain: {selected_row['Gain (dBi)']} dBi")
                st.write(f"Default Tx Power: {default_power:.2f} dBm")
                st.write(f"Adjusted Tx Power: {power_adjustment:.2f} dBm")
                st.write(f"Power Adjustment: {power_adjustment - default_power:.2f} dB")
                st.write(
                    f"Adjusted Gain: {float(selected_row['Gain (dBi)']) + (power_adjustment - default_power):.2f} dBi"
                )
                st.write(f"Manufacturer: {selected_row['Manufacturer']}")
                st.write(
                    f"Pattern Electrical Tilt: {selected_row['Pattern Electrical Tilt (°)']}°"
                )
                st.write(
                    f"Half-power Beamwidth: {selected_row['Half-power Beamwidth']}°"
                )
                st.write(
                    f"Frequency Range: {selected_row['Min Frequency (MHz)']} - {selected_row['Max Frequency (MHz)']} MHz"
                )
                st.write(
                    f"Pattern Electrical Azimuth: {selected_row['Pattern Electrical Azimuth (°)']}°"
                )
                st.write(f"Comments: {selected_row['Comments']}")

            else:
                st.error("Could not extract pattern data for the selected comment.")
        elif df is not None:
            st.error(
                'The uploaded file does not contain the expected "Comments" or "Max Frequency (MHz)" column.'
            )


if __name__ == "__main__":
    main()
