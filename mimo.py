import colorsys
import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    @staticmethod
    def _hsv_to_hex(hue: float) -> str:
        rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        return f"#{int(rgb[0] * 255):02x}{int(rgb[1] * 255):02x}{int(rgb[2] * 255):02x}"


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


def create_combined_polar_chart(patterns: list[AntennaPattern], name: str):
    """Create a combined polar chart for horizontal and vertical patterns."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(16, 8), subplot_kw={"projection": "polar"}
    )
    fig.suptitle(f"Antenna Patterns for {name}", fontsize=16)

    for i, pattern in enumerate(patterns):
        hue = i / len(patterns)
        color = AntennaPattern._hsv_to_hex(hue)

        # Horizontal pattern
        normalized_gain_h = 10 ** (-pattern.horizontal["Att dB"] / 20)
        ax1.plot(
            np.radians(pattern.horizontal["angle"]),
            normalized_gain_h,
            color=color,
        )

        # Vertical pattern
        normalized_gain_v = 10 ** (-pattern.vertical["Att dB"] / 20)
        ax2.plot(
            np.radians(pattern.vertical["angle"]),
            normalized_gain_v,
            color=color,
        )

    for ax in (ax1, ax2):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_yticklabels(["0", "-14", "-8", "-4", "-2", "0"])
        ax.grid(True)
        ax.legend(loc="lower left", bbox_to_anchor=(0.1, -0.1))

    ax1.set_title("Horizontal Pattern")
    ax2.set_title("Vertical Pattern")

    return fig


def main():
    st.set_page_config(page_title="Antenna Pattern", layout="wide", page_icon="🧊")

    # Define the path to the fixed file
    script_dir = os.path.dirname(__file__)
    fixed_file_path = os.path.join(script_dir, "AIR6488.txt")

    # Load and cache the file content
    if "df" not in st.session_state:
        st.session_state.df = decode_file_content(fixed_file_path)

    if st.session_state.df is not None:
        if "Name" in st.session_state.df.columns:
            names = st.session_state.df["Name"].unique()
            selected_name = st.selectbox("Select Name", names, key="selected_name")

            selected_df = st.session_state.df[
                st.session_state.df["Name"] == selected_name
            ].copy()
            selected_df.sort_values(
                by="Pattern Electrical Azimuth (°)", ascending=True, inplace=True
            )

            patterns = []
            for _, row in selected_df.iterrows():
                horizontal_pattern, vertical_pattern = extract_pattern(row["Pattern"])
                if horizontal_pattern and vertical_pattern:
                    antenna_params = extract_antenna_parameters(row["Comments"])

                    pattern = AntennaPattern(
                        horizontal=transpose_pattern(horizontal_pattern),
                        vertical=transpose_pattern(vertical_pattern),
                        gain=float(row["Gain (dBi)"]),
                        tilt=float(row["Pattern Electrical Tilt (°)"]),
                        beamwidth=float(row["Half-power Beamwidth"]),
                        frequency=float(row["Max Frequency (MHz)"]),
                        **antenna_params,
                    )
                    patterns.append(pattern)

            if patterns:
                fig = create_combined_polar_chart(patterns, selected_name)
                st.pyplot(fig)

                # Create a DataFrame for the antenna parameters
                param_data = []
                for i, pattern in enumerate(patterns):
                    param_data.append(
                        {
                            "Beam Index": i + 1,
                            "Gain (dBi)": f"{pattern.gain:.2f}",
                            "Tilt (°)": f"{pattern.tilt:.2f}",
                            "H-Beamwidth (°)": f"{pattern.h_beamwidth:.2f}",
                            "V-Beamwidth (°)": f"{pattern.v_beamwidth:.2f}",
                            "Frequency (MHz)": f"{pattern.frequency:.2f}",
                            "Tx Power (dBm)": f"{pattern.tx_power:.2f}",
                        }
                    )
                param_df = pd.DataFrame(param_data)

                # Display the antenna parameters table
                st.table(param_df)

            else:
                st.warning("No valid patterns found for the selected antenna.")
        else:
            st.error('The uploaded file does not contain the expected "Name" column.')
    else:
        st.error("Failed to load the antenna data file.")


if __name__ == "__main__":
    main()
