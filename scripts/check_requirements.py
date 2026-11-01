"""Check whether every direct requirement is installed at a compatible version."""
from __future__ import annotations

from importlib import metadata
from pathlib import Path
import sys

from packaging.requirements import Requirement


def missing_requirements(requirements_file: Path) -> list[str]:
    missing: list[str] = []
    for raw_line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker and not requirement.marker.evaluate():
            continue
        try:
            installed = metadata.version(requirement.name)
        except metadata.PackageNotFoundError:
            missing.append(f"{requirement.name} is not installed")
            continue
        if requirement.specifier and not requirement.specifier.contains(installed, prereleases=True):
            missing.append(f"{requirement.name} {installed} does not satisfy {requirement.specifier}")
    return missing


def main() -> int:
    requirements_file = Path(__file__).resolve().parents[1] / "requirements.txt"
    missing = missing_requirements(requirements_file)
    if missing:
        print("Required packages need installation or repair:")
        for item in missing:
            print(f"  - {item}")
        return 1
    print("All direct requirements are installed at compatible versions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
