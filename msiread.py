import numpy as np


def msi_read(fname):
    """
    Read MSI planet Antenna file (.pln, .msi).

    Parameters
    ----------
    fname (str): Filename with extension .pln or .msi

    Returns
    -------
    tuple: Contains up to three elements:
        - horizontal: Dictionary containing horizontal gain data
        - vertical: Dictionary containing vertical gain data
        - optional: Dictionary containing additional fields from the file
    """

    def initialize_data():
        """Initialize the data structures"""
        horizontal = {
            "PhysicalQuantity": None,
            "Magnitude": None,
            "Units": None,
            "Azimuth": None,
            "Elevation": None,
            "Frequency": None,
            "Slice": None,
        }

        vertical = horizontal.copy()
        optional = {}
        return horizontal, vertical, optional

    def reduce_spaces(text):
        """Reduce contiguous spaces to single spaces and remove leading/trailing spaces"""
        if not isinstance(text, str):
            return text

        # Remove leading/trailing spaces
        text = text.strip()

        # Reduce multiple spaces to single space
        while "  " in text:
            text = text.replace("  ", " ")

        return text

    def get_scalar_numeric_value(txt):
        """Convert string to scalar numeric value"""
        try:
            values = [float(x) for x in txt.strip().split()]
            if len(values) != 1:
                return None, "Format error - too many numeric values found"
            return values[0], ""
        except ValueError:
            return None, "Not a numeric value"

    def get_scalar_numeric_value_or_string(txt):
        """Convert string to either numeric value or string"""
        value, msg = get_scalar_numeric_value(txt)
        if msg:
            return txt.strip(), ""
        return value, ""

    def get_value_with_unit(txt):
        """Return numeric value and string unit"""
        parts = txt.strip().split()

        if len(parts) == 0:
            return None, None, "No value found"

        try:
            value = float(parts[0])
        except ValueError:
            return None, None, "First value must be numeric"

        unit = "dBd" if len(parts) == 1 else " ".join(parts[1:])
        return value, unit, ""

    def get_xy_coords(file):
        """Parse next text lines for X Y coordinate pairs"""
        coords = []

        while True:
            line = file.readline()
            if not line:  # EOF
                return np.array(coords), "", "", None

            line = line.strip()
            if not line:  # Skip blank lines
                continue

            line = reduce_spaces(line)
            parts = line.split()

            if len(parts) < 2:
                # Check if it's a new keyword
                try:
                    float(parts[0])
                    return (
                        None,
                        "Format error - a frequency without a value was found",
                        "",
                        None,
                    )
                except ValueError:
                    return np.array(coords), "", line, None

            try:
                x, y = float(parts[0]), float(parts[1])
                coords.append([x, y])
            except ValueError:
                return np.array(coords), "", line, None

    # Initialize data structures
    horizontal, vertical, optional = initialize_data()

    try:
        with open(fname) as file:
            parse_next_line = True
            line_num = 0

            while True:
                if parse_next_line:
                    line = file.readline()
                    if not line:  # EOF
                        break

                    line_num += 1
                    line = reduce_spaces(line)

                parts = line.split(maxsplit=1)
                if not parts:
                    continue

                first_word = parts[0].lower()
                rest_of_line = parts[1] if len(parts) > 1 else ""

                if first_word in ["name", "make"]:
                    optional[first_word] = rest_of_line.strip()

                elif first_word in ["h_width", "v_width", "front_to_back"]:
                    value, msg = get_scalar_numeric_value(rest_of_line)
                    if msg:
                        raise ValueError(f"Error on line {line_num}: {msg}")
                    optional[first_word] = value

                elif first_word == "frequency":
                    value, msg = get_scalar_numeric_value(rest_of_line)
                    if msg:
                        raise ValueError(f"Error on line {line_num}: {msg}")
                    freq = value * 1e6
                    horizontal["Frequency"] = freq
                    vertical["Frequency"] = freq
                    optional[first_word] = freq

                elif first_word == "tilt":
                    value, msg = get_scalar_numeric_value_or_string(rest_of_line)
                    optional[first_word] = value

                elif first_word == "gain":
                    value, unit, msg = get_value_with_unit(rest_of_line)
                    if msg:
                        raise ValueError(f"Error on line {line_num}: {msg}")
                    optional[first_word] = {"value": value, "unit": unit}

                elif first_word == "polarization":
                    optional[first_word] = rest_of_line.strip()

                elif first_word == "comment":
                    optional[first_word] = rest_of_line.strip()

                elif first_word in ["horizontal", "vertical"]:
                    count, msg = get_scalar_numeric_value(rest_of_line)
                    if msg:
                        raise ValueError(f"Error on line {line_num}: {msg}")

                    data, msg, next_line, _ = get_xy_coords(file)
                    if msg:
                        raise ValueError(f"Error reading coordinates: {msg}")

                    if data is not None and len(data) > 0:
                        trans_data = (-1 * data[:, 1]) + optional["gain"]["value"]

                        if first_word == "horizontal":
                            horizontal["Azimuth"] = data[:, 0]
                            horizontal["Magnitude"] = trans_data
                            horizontal["Elevation"] = 0
                            horizontal["Slice"] = "Elevation"
                        else:
                            vertical["Elevation"] = data[:, 0]
                            vertical["Magnitude"] = trans_data
                            vertical["Azimuth"] = 0
                            vertical["Slice"] = "Azimuth"

                        if count != len(data):
                            print(
                                f"Warning: {first_word} data count in file ({count}) does not match number of data elements ({len(data)})"
                            )

                    parse_next_line = not bool(next_line)
                    if next_line:
                        line = next_line

                else:
                    optional[first_word] = rest_of_line.strip()

        # Set units based on gain
        if optional.get("gain", {}).get("unit") == "dBi":
            horizontal["Units"] = "dBi"
            vertical["Units"] = "dBi"

        return horizontal, vertical, optional

    except Exception as e:
        raise RuntimeError(f"Error reading file: {e!s}")
