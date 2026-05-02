def create_regex_for_prefix_match(prefixes):
    """
    Creates a regex pattern that matches any string starting with any of the provided prefixes.

    Parameters:
    - prefixes (list): A list of prefixes to match at the start of a string.

    Returns:
    - str: A regex pattern string.
    """
    return '^(' + '|'.join(prefixes) + ')'