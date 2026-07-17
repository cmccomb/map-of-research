# Canonical faculty registry

`faculty.csv` is the sole source of truth for map membership. Each row contains
a map slug, its human-readable department name, the label used on that map, and
an optional public Scholar profile ID.

Rows without a Scholar ID are retained for visibility but are not collected.
Repeated Scholar IDs are expected for cross-appointed faculty and are fetched
only once.
