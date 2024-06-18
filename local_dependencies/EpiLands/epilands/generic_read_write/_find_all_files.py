import os
import logging
import re
from typing import Union

sub_package_name = os.path.split(os.path.dirname(os.path.abspath(__file__)))[1]
logger = logging.getLogger(sub_package_name)


def find_all_files(
    path: str, pattern: Union[str, re.Pattern], regex: bool = False
) -> list:
    """
    Recursively searches for all files in a given path that match a given pattern.

    Parameters:
    -----------
    path : str
        The path to search for files.
    pattern : Union[str, re.Pattern]
        The pattern to match the files against. Can be a string or a regular expression pattern.
    regex : bool, optional
        If True, pattern is treated as a regular expression pattern. Default is False.

    Returns:
    --------
    list
        A list of file paths that match the pattern.

    Raises:
    -------
    None

    Examples:
    ---------
    Find all files in a directory that contain the word "example":

    >>> find_all_files("/path/to/directory", "example")

    Find all files in a directory that match a regular expression:

    >>> import re
    >>> pattern = re.compile("^example_.*\.txt$")
    >>> find_all_files("/path/to/directory", pattern, regex=True)
    """

    if isinstance(pattern, re.Pattern):
        regex = True
    if regex == True and isinstance(pattern, str):
        pattern = re.compile(pattern)
    file_paths = []

    search_msg = f"keyword {pattern}" if regex == False else f"regex {pattern}"
    logger.info(f"Searching in {path} for {search_msg}")
    for dirpath, dirnames, filenames in os.walk(path):
        num_already_found = len(file_paths)
        for filename in filenames:
            if regex == False and pattern in filename:
                # logger.debug(f"found {filename} in {dirpath}")
                file_paths.append(os.path.join(dirpath, filename))
            elif regex == True and re.search(pattern, filename):
                # logger.debug(f"found {filename} in {dirpath}")
                file_paths.append(os.path.join(dirpath, filename))
        logger.debug(f"Found {len(file_paths)-num_already_found} files in {dirpath}")
    if len(file_paths) == 0:
        logger.warning(f"No files found in {path}")
    else:
        logger.info(f"Found {len(file_paths)} files in {path}")
    return file_paths
