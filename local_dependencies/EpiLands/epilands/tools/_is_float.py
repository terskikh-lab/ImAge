# See https://stackoverflow.com/questions/54729156/how-to-check-if-a-string-is-a-decimal-float-number
import re


def is_float(item):
    # A float is a float
    if isinstance(item, float):
        return True

    # Ints are okay
    if isinstance(item, int):
        return True

    # Detect leading white-spaces
    if len(item) != len(item.strip()):
        return False

    # Some strings can represent floats or ints ( i.e. a decimal )
    if isinstance(item, str):
        # regex matching
        int_pattern = re.compile("^[0-9]*$")
        float_pattern = re.compile("^[0-9]*.[0-9]*$")
        if float_pattern.match(item) or int_pattern.match(item):
            return True
        else:
            return False
