import numpy as np
import plotly.graph_objects as go
import streamlit as st

from msiread import msi_read
from patternFromSlices import pattern_from_slices


def create_polar_plot(angles, magnitudes, title):
    """Create a polar plot using Plotly"""
    fig = go.Figure()

    # Convert angles to radians for plotting
    theta = np.deg2rad(angles)

    # Add the polar trace
    fig.add_trace(
        go.Scatterpolar(
            r=magnitudes,
            theta=angles,
            mode="lines",
            name="Antenna Pattern",
            line=dict(color="blue"),
        )
    )

    # Update layout
    fig.update_layout(
        title=title,
        showlegend=True,
        polar=dict(
            radialaxis=dict(visible=True, range=[min(magnitudes), max(magnitudes)])
        ),
    )

    return fig


def create_3d_pattern(pattern_data, theta, phi):
    """Create 3D surface plot using Plotly"""
    # Create meshgrid for plotting
    THETA, PHI = np.meshgrid(theta, phi)

    # Convert spherical coordinates to Cartesian
    X = pattern_data * np.sin(np.deg2rad(THETA)) * np.cos(np.deg2rad(PHI))
    Y = pattern_data * np.sin(np.deg2rad(THETA)) * np.sin(np.deg2rad(PHI))
    Z = pattern_data * np.cos(np.deg2rad(THETA))

    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])

    fig.update_layout(
        title="3D Antenna Pattern",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5),
            ),
        ),
    )

    return fig


def main():
    st.title("Antenna Pattern Visualization")

    # File uploader
    uploaded_file = st.file_uploader("Upload MSI file", type=["msi"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp.msi", "wb") as f:
            f.write(uploaded_file.getvalue())

        try:
            # Read MSI data
            Horizontal, Vertical, Optional = msi_read("temp.msi")

            # Display basic information
            st.subheader("Antenna Parameters")
            st.write(f"Frequency: {Optional.get('frequency', 'N/A')} Hz")
            st.write(
                f"Gain: {Optional.get('gain', {}).get('value', 'N/A')} {Optional.get('gain', {}).get('unit', '')}"
            )

            # Normalize the azimuth to 0 degrees
            # Horizontal["Azimuth"] = np.mod(Horizontal["Azimuth"], 360)
            # Horizontal["Azimuth"] = Horizontal["Azimuth"] - Horizontal["Azimuth"][0]

            # Create and display vertical pattern polar plot
            st.subheader("Vertical Pattern")
            vertical_fig = create_polar_plot(
                Vertical["Elevation"], Vertical["Magnitude"], "Vertical Pattern"
            )
            st.plotly_chart(vertical_fig)

            # Create and display horizontal pattern polar plot
            st.subheader("Horizontal Pattern")
            horizontal_fig = create_polar_plot(
                Horizontal["Azimuth"], Horizontal["Magnitude"], "Horizontal Pattern"
            )
            st.plotly_chart(horizontal_fig)

            # Pattern reconstruction for 3D visualization
            st.subheader("3D Pattern")

            # Convert elevation angles to theta
            theta = 90 - np.array(Vertical["Elevation"])
            # st.write("theta", theta)
            phi = np.array(Horizontal["Azimuth"])
            # st.write("phi", phi)

            # Ensure we have the correct number of points (73 points as in the example)
            theta_interp = np.linspace(min(theta), max(theta), 73)
            # st.write("theta_interp", theta_interp)
            phi_interp = np.linspace(min(phi), max(phi), 73)
            # st.write("phi_interp", phi_interp)
            # Interpolate the slices to match the number of points
            vert_slice_interp = np.interp(theta_interp, theta, Vertical["Magnitude"])
            # st.write(vert_slice_interp)
            horiz_slice_interp = np.interp(phi_interp, phi, Horizontal["Magnitude"])
            # st.write(horiz_slice_interp)
            # Debug statements in main function
            st.write("Vertical Elevation", Vertical["Elevation"])
            st.write("Vertical Magnitude", Vertical["Magnitude"])
            st.write("Horizontal Azimuth", Horizontal["Azimuth"])
            st.write("Horizontal Magnitude", Horizontal["Magnitude"])

            # Interpolation Debug
            st.write("Interpolated Theta", theta_interp)
            st.write("Interpolated Phi", phi_interp)
            st.write("Interpolated Vertical Slice", vert_slice_interp)
            st.write("Interpolated Horizontal Slice", horiz_slice_interp)
            # Generate 3D pattern using pattern_from_slices
            pattern_3d, theta_out, phi_out = pattern_from_slices(
                vert_slice=vert_slice_interp,
                theta=theta_interp,
                horiz_slice=horiz_slice_interp,
                phi=phi_interp,
                method="Summing",
            )
            st.write("Pattern 3D Shape", pattern_3d.shape)
            st.write("Pattern 3D Min/Max", np.min(pattern_3d), np.max(pattern_3d))

            # Create and display 3D pattern
            pattern_3d_fig = create_3d_pattern(pattern_3d, theta_out, phi_out)
            st.plotly_chart(pattern_3d_fig)

            # Add download button for data
            if st.button("Download Pattern Data"):
                # Create download links for the data
                np.savez(
                    "pattern_data.npz",
                    pattern_3d=pattern_3d,
                    theta=theta_out,
                    phi=phi_out,
                    vertical_elevation=Vertical["Elevation"],
                    vertical_magnitude=Vertical["Magnitude"],
                    horizontal_azimuth=Horizontal["Azimuth"],
                    horizontal_magnitude=Horizontal["Magnitude"],
                )
                st.download_button(
                    label="Download Pattern Data (NPZ)",
                    data=open("pattern_data.npz", "rb"),
                    file_name="pattern_data.npz",
                    mime="application/octet-stream",
                )

        except Exception as e:
            st.error(f"Error processing file: {e!s}")


if __name__ == "__main__":
    main()
