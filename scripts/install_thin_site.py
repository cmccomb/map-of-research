#!/usr/bin/env python3
"""Install the canonical dependency-free map client into one site repository."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SITE_TEMPLATE = PROJECT_ROOT / "site"


def install_site(
    repository: Path,
    *,
    map_slug: str,
    title: str,
    remove_legacy: bool = False,
) -> None:
    if not (repository / ".git").is_dir():
        raise ValueError(f"Target is not a Git repository: {repository}")
    if repository.name != map_slug:
        raise ValueError(
            f"Repository name {repository.name!r} does not match {map_slug!r}"
        )
    assets = repository / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SITE_TEMPLATE / "index.html", repository / "index.html")
    shutil.copyfile(SITE_TEMPLATE / "assets/map.css", assets / "map.css")
    shutil.copyfile(SITE_TEMPLATE / "assets/map.js", assets / "map.js")
    shutil.copyfile(SITE_TEMPLATE / "assets/favicon.svg", assets / "favicon.svg")
    (repository / "map-config.json").write_text(
        json.dumps(
            {
                "map_slug": map_slug,
                "title": title,
                "dataset_id": "ccm/cmu-engineering-publications",
                "dataset_revision": "main",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    readme = (SITE_TEMPLATE / "README.template.md").read_text(encoding="utf-8")
    (repository / "README.md").write_text(
        readme.replace("{{TITLE}}", title).replace("{{SLUG}}", map_slug),
        encoding="utf-8",
    )
    shutil.copyfile(SITE_TEMPLATE / "SECURITY.md", repository / "SECURITY.md")
    workflows = repository / ".github/workflows"
    workflows.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        SITE_TEMPLATE / ".github/workflows/ci.yml",
        workflows / "ci.yml",
    )
    shutil.copyfile(
        SITE_TEMPLATE / ".github/workflows/deploy.yml",
        workflows / "deploy.yml",
    )
    dependabot = repository / ".github/dependabot.yml"
    dependabot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SITE_TEMPLATE / ".github/dependabot.yml", dependabot)

    if remove_legacy:
        for workflow_name in (
            "static.yml",
            "update.yml",
            "update_embeddings.yml",
            "update_visualization.yml",
        ):
            (workflows / workflow_name).unlink(missing_ok=True)
        for file_name in (
            "data.csv",
            "faculty.csv",
            "requirements.txt",
            "update_embeddings.py",
            "update_visualization.py",
        ):
            (repository / file_name).unlink(missing_ok=True)
        for directory_name in ("data", "data_old"):
            directory = repository / directory_name
            if directory.is_dir():
                shutil.rmtree(directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--map-slug", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument(
        "--remove-legacy",
        action="store_true",
        help="Remove repository-local scraping, generated data, and old workflows",
    )
    args = parser.parse_args()
    install_site(
        args.repository,
        map_slug=args.map_slug,
        title=args.title,
        remove_legacy=args.remove_legacy,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
