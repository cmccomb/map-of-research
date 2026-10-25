"""Fail closed when an included Scholar lacks an evidenced primary home."""

from __future__ import annotations

import argparse
from pathlib import Path

from map_of_research.registry import (
    DEFAULT_DEPARTMENTS_PATH,
    DEFAULT_IMPLICIT_PRIMARY_BASELINE_PATH,
    DEFAULT_MEMBERSHIPS_PATH,
    DEFAULT_PEOPLE_PATH,
    load_implicit_primary_baseline,
    load_registry,
    validate_included_scholar_primary_homes,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people", type=Path, default=DEFAULT_PEOPLE_PATH)
    parser.add_argument("--memberships", type=Path, default=DEFAULT_MEMBERSHIPS_PATH)
    parser.add_argument("--departments", type=Path, default=DEFAULT_DEPARTMENTS_PATH)
    parser.add_argument(
        "--baseline", type=Path, default=DEFAULT_IMPLICIT_PRIMARY_BASELINE_PATH
    )
    args = parser.parse_args()
    registry = load_registry(args.people, args.memberships, args.departments)
    count = validate_included_scholar_primary_homes(
        registry, legacy_implicit_primary=load_implicit_primary_baseline(args.baseline)
    )
    print(f"Validated {count} included Scholar primary academic homes")


if __name__ == "__main__":
    main()
