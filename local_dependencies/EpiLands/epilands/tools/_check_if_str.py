def check_if_str(s) -> str:
    if isinstance(s, str):
        return s
    if not isinstance(s, str):
        return str(s)
