"""Run the controlled private capture-to-public subset audit before release.

The full captures and reviews remain outside source Git. The output receipt has
only hashes and aggregate counts, so it can be pinned by the release gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from map_of_research.private_subset_preflight import validate_private_public_subsets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--private-outputs", type=Path, required=True)
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(args.private_outputs.resolve()):
        parser.error(
            "The audit receipt must be beside private evidence, outside source Git"
        )
    receipt = validate_private_public_subsets(
        source_root=args.source_root,
        private_outputs=args.private_outputs,
        selection_manifest_path=args.selection_manifest,
        snapshot_path=args.snapshot,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"status": receipt["status"], "totals": receipt["totals"]}, sort_keys=True
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
