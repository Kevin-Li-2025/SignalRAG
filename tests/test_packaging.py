import tomllib
from pathlib import Path


def test_pyproject_limits_setuptools_package_discovery() -> None:
    data = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert data["tool"]["setuptools"]["packages"]["find"]["include"] == [
        "fast_rag*"
    ]
    assert data["tool"]["setuptools"]["package-data"]["fast_rag"] == ["static/*"]
