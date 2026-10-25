"""Pin the last successful hosted release before a new upload starts."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Sequence
from pathlib import Path

from .region_continuity import require_revision


def pin_prior_release(status_dir: Path) -> str:
    """Return the immutable prior SHA and save the first validated receipt."""

    successful = status_dir / "last-successful-upload.json"
    source = successful if successful.exists() else status_dir / "last-upload.json"
    receipt = json.loads(source.read_text(encoding="utf-8"))
    if receipt.get("status") != "success":
        raise ValueError(f"Prior hosted release receipt is not successful: {source}")
    revision = require_revision(str(receipt.get("artifact_commit") or ""))
    if not successful.exists():
        shutil.copyfile(source, successful)
    return revision


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(f"revision={pin_prior_release(args.status_dir)}")


if __name__ == "__main__":
    main()
