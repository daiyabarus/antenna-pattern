def msiread(fname):
    """
    Reads an MSI Planet Antenna file (.pln, .msi).

    Returns a tuple of dictionaries: (Horizontal, Vertical, Optional).
    """

    def reduce_spaces(s):
        return " ".join(s.split())

    def get_scalar_numeric_value(txt, line_num):
        try:
            return float(txt.strip()), ""
        except ValueError:
            return None, f"Format error - no numeric value found on line {line_num}"

    def get_value_with_unit(txt, line_num):
        parts = txt.split()
        value, msg = get_scalar_numeric_value(parts[0], line_num)
        unit = parts[1] if len(parts) > 1 else "dBd"
        return value, unit, msg

    def get_xy_coords(lines, line_num):
        vec = []
        for line in lines:
            line_num += 1
            line = reduce_spaces(line.strip())
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                return (
                    vec,
                    f"Format error - expected 2 values on line {line_num}",
                    line_num,
                )
            try:
                vec.append([float(parts[0]), float(parts[1])])
            except ValueError:
                return vec, "", line_num
        return vec, "", line_num

    Horizontal = {
        key: None
        for key in [
            "PhysicalQuantity",
            "Magnitude",
            "Units",
            "Azimuth",
            "Elevation",
            "Frequency",
            "Slice",
        ]
    }
    Vertical = {
        key: None
        for key in [
            "PhysicalQuantity",
            "Magnitude",
            "Units",
            "Azimuth",
            "Elevation",
            "Frequency",
            "Slice",
        ]
    }
    Optional = {}

    try:
        with open(fname) as file:
            lines = file.readlines()

        line_num = 0
        optional_gain_value = None

        while line_num < len(lines):
            line = reduce_spaces(lines[line_num].strip())
            line_num += 1
            if not line:
                continue

            first_word, rest_of_line = line.split(" ", 1) if " " in line else (line, "")
            first_word = first_word.lower()

            if first_word in [
                "name",
                "make",
                "h_width",
                "v_width",
                "front_to_back",
                "frequency",
                "tilt",
                "gain",
                "polarization",
                "comment",
            ]:
                if first_word in ["h_width", "v_width", "front_to_back", "frequency"]:
                    value, _ = get_scalar_numeric_value(rest_of_line, line_num)
                    Optional[first_word] = value
                    if first_word == "frequency":
                        Horizontal["Frequency"] = Vertical["Frequency"] = value * 1e6
                elif first_word == "tilt":
                    Optional[first_word] = rest_of_line.strip()
                elif first_word == "gain":
                    value, unit, _ = get_value_with_unit(rest_of_line, line_num)
                    Optional[first_word] = {"value": value, "unit": unit}
                    optional_gain_value = value
                else:
                    Optional[first_word] = rest_of_line.strip()
            elif first_word in ["horizontal", "vertical"]:
                N, _ = get_scalar_numeric_value(rest_of_line, line_num)
                data, _, line_num = get_xy_coords(lines[line_num:], line_num)
                TransData = (
                    [-1 * d[1] + optional_gain_value for d in data]
                    if optional_gain_value is not None
                    else [d[1] for d in data]
                )
                if first_word == "horizontal":
                    Horizontal["Azimuth"], Horizontal["Magnitude"] = [
                        d[0] for d in data
                    ], TransData
                else:
                    Vertical["Azimuth"], Vertical["Magnitude"] = [
                        d[0] for d in data
                    ], TransData
            else:
                Optional[first_word] = rest_of_line.strip()

        if "gain" in Optional and Optional["gain"]["unit"] == "dBi":
            Horizontal["Units"] = Vertical["Units"] = "dBi"

        return Horizontal, Vertical, Optional

    except Exception as e:
        raise RuntimeError(f"Error reading file {fname}: {e!s}")
