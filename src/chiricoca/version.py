import tomllib
from pathlib import Path


def get_version():
    path = Path(__file__).parent.parent.parent / "pyproject.toml"
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


__version__ = get_version()
