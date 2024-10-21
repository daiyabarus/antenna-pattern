def msiread(fname):
    """
    Reads an MSI Planet Antenna file (.pln, .msi).

    Returns a tuple of dictionaries: (Horizontal, Vertical, Optional).
    """

    def reduce_spaces(s):
        return " ".join(s.split())

    def get_scalar_numeric_value(txt, line_num):
        try:
            value = float(txt.strip())
            return value, ""
        except ValueError:
            return None, f"Format error - no numeric value found on line {line_num}"

    def get_value_with_unit(txt, line_num):
        parts = txt.split()
        if len(parts) == 1:
            value, msg = get_scalar_numeric_value(parts[0], line_num)
            unit = "dBd"
        else:
            value, msg = get_scalar_numeric_value(parts[0], line_num)
            unit = parts[1]
        return value, unit, msg

    def get_scalar_numeric_value_or_string(txt, line_num):
        try:
            value, msg = get_scalar_numeric_value(txt, line_num)
            if msg:
                return txt.strip(), ""
            return value, ""
        except ValueError:
            return txt.strip(), ""

    def get_one_string_value(txt):
        return txt.strip(), ""

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
                x, y = float(parts[0]), float(parts[1])
                vec.append([x, y])
            except ValueError:
                return vec, "", line_num
        return vec, "", line_num

    # Initialize output structures
    Horizontal = {
        "PhysicalQuantity": None,
        "Magnitude": None,
        "Units": None,
        "Azimuth": None,
        "Elevation": None,
        "Frequency": None,
        "Slice": None,
    }
    Vertical = {
        "PhysicalQuantity": None,
        "Magnitude": None,
        "Units": None,
        "Azimuth": None,
        "Elevation": None,
        "Frequency": None,
        "Slice": None,
    }
    Optional = {}

    try:
        with open(fname) as file:
            lines = file.readlines()

        parse_next_line = True
        line_num = 0
        n, m = 0, 0
        optional_gain_value = None

        while line_num < len(lines):
            if parse_next_line:
                line = lines[line_num].strip()
                line_num += 1
                if not line:
                    continue
                line = reduce_spaces(line)

            idx = line.find(" ")
            if idx == -1:
                idx = len(line)

            first_word = line[:idx].lower()
            rest_of_line = line[idx:].strip()

            if first_word in ["name", "make"]:
                Optional[first_word] = rest_of_line
            elif first_word in ["h_width", "v_width", "front_to_back", "frequency"]:
                value, msg = get_scalar_numeric_value(rest_of_line, line_num)
                Optional[first_word] = value
                if first_word == "frequency":
                    Horizontal["Frequency"] = value * 1e6
                    Vertical["Frequency"] = value * 1e6
            elif first_word == "tilt":
                value, _ = get_scalar_numeric_value_or_string(rest_of_line, line_num)
                Optional[first_word] = value
            elif first_word == "gain":
                value, unit, msg = get_value_with_unit(rest_of_line, line_num)
                Optional[first_word] = {"value": value, "unit": unit}
                optional_gain_value = value
            elif first_word == "polarization":
                Optional[first_word], _ = get_one_string_value(rest_of_line)
            elif first_word == "comment":
                Optional[first_word] = rest_of_line
            elif first_word in ["horizontal", "vertical"]:
                N, msg = get_scalar_numeric_value(rest_of_line, line_num)
                if first_word == "horizontal":
                    n += 1
                    data, msg, line_num = get_xy_coords(lines[line_num:], line_num)
                    if optional_gain_value is not None:
                        TransData = [-1 * d[1] + optional_gain_value for d in data]
                    Horizontal["Azimuth"] = [d[0] for d in data]
                    Horizontal["Magnitude"] = TransData
                elif first_word == "vertical":
                    m += 1
                    data, msg, line_num = get_xy_coords(lines[line_num:], line_num)
                    if optional_gain_value is not None:
                        TransData = [-1 * d[1] + optional_gain_value for d in data]
                    Vertical["Azimuth"] = [d[0] for d in data]
                    Vertical["Magnitude"] = TransData
            else:
                Optional[first_word] = rest_of_line

        if "gain" in Optional and Optional["gain"]["unit"] == "dBi":
            Horizontal["Units"] = "dBi"
            Vertical["Units"] = "dBi"

        return Horizontal, Vertical, Optional

    except Exception as e:
        raise RuntimeError(f"Error reading file {fname}: {e!s}")
