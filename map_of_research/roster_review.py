"""Build the annual, human-verified official faculty roster review."""

from __future__ import annotations

import argparse
import datetime as dt
from collections import Counter
from pathlib import Path

from .registry import load_registry


def build_review(*, today: dt.date | None = None) -> str:
    """Return a complete annual review issue body without changing the registry."""

    registry = load_registry()
    review_date = today or dt.datetime.now(dt.UTC).date()
    people = registry.people_by_id
    memberships_by_department = {
        department.department_id: [
            membership
            for membership in registry.memberships
            if membership.department_id == department.department_id
        ]
        for department in registry.departments
    }
    lines = [
        f"# Annual faculty roster review — {review_date.year}",
        "",
        "Review each official CMU directory for newly added faculty and role changes.",
        "Do not edit Scholar data automatically or remove historical people records.",
        "",
        "## Inclusion policy",
        "",
        "- Include current faculty, teaching faculty, and emeriti.",
        "- Retain but exclude affiliate, courtesy, visiting, former, and other roles.",
        "- Record new people even when no Google Scholar profile can be verified.",
        "- Verify Scholar IDs through an institutional, faculty-controlled, or "
        "self-managed public link; never bulk-search Scholar.",
        "- Record the verification URL and date in `registry/people.csv`.",
        "- After review, update `verified_at` on memberships and `reviewed_at` "
        "in `registry/departments.csv`.",
        "",
        "## Department checklist",
        "",
        "| Done | Department or program | Included | Retained/excluded | "
        "Missing Scholar ID | Last review | Official directory |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for department in registry.departments:
        memberships = memberships_by_department[department.department_id]
        included = [membership for membership in memberships if membership.included]
        excluded = [membership for membership in memberships if not membership.included]
        missing_ids = sum(
            1 for membership in included if not people[membership.person_id].scholar_id
        )
        lines.append(
            "| [ ] | "
            f"{department.title} | {len(included)} | {len(excluded)} | "
            f"{missing_ids} | {department.reviewed_at or 'Never'} | "
            f"[Open directory]({department.directory_url}) |"
        )

    lines.extend(["", "## Current role summary", ""])
    role_counts = Counter(membership.role for membership in registry.memberships)
    for role, count in sorted(role_counts.items()):
        lines.append(f"- `{role}`: {count}")

    missing_people = [
        person
        for person in registry.people
        if not person.scholar_id
        and any(
            membership.included and membership.person_id == person.person_id
            for membership in registry.memberships
        )
    ]
    lines.extend(
        [
            "",
            f"## Included people without a verified Scholar ID ({len(missing_people)})",
            "",
        ]
    )
    lines.extend(
        f"- {person.display_name} (`{person.person_id}`)" for person in missing_people
    )
    lines.extend(
        [
            "",
            "## Completion checks",
            "",
            "- [ ] Every official directory was reviewed.",
            "- [ ] New faculty and teaching faculty were added.",
            "- [ ] New emeriti were retained with role `emeritus`.",
            "- [ ] Departed faculty were retained with role `former` and excluded.",
            "- [ ] Affiliates and courtesy appointments remain excluded.",
            "- [ ] Scholar IDs have verification URLs and dates; unresolved IDs "
            "remain blank.",
            "- [ ] Registry tests and a dry-run collection passed.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    review = build_review()
    if args.output:
        args.output.write_text(review, encoding="utf-8")
    else:
        print(review)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
