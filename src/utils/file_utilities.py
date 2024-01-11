import os


def scan_directory(directory: str, extension: str) -> list:
    """Check specified directory and return list of files with
    specified extension

    Args:
        directory (str): path string to directory e.g. "./the/directory"
        extension (str): extension type to be searched for e.g. ".csv"

    Returns:
        list: strings of file names with specified extension
    """

    files: list = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            files.append(filename)
    return files
