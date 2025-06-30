import json

def load_json(path: str) -> dict:
    """
    Loads and returns the JSON content from the given file.

    Args:
        path: Filesystem path to the JSON state file.

    Returns:
        A dictionary parsed from the JSON file.
    """
    with open(path, 'r') as f:
        return json.load(f)