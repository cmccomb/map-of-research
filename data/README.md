# Collector cache

The `automation/map-snapshot` branch stores one atomic JSON cache file per
unique Scholar profile under `data/authors/`. The default branch owns the
trusted pipeline code and source definitions.

Cache records retain the last good normalized source observation. Registry role
changes never delete caches; excluded and former relationships remain available
to the lossless dataset pipeline.

For new direct browser captures, store matching HTML and extracted row JSON in
`data/captures/`. The Fonseca completion includes both files; its cache names
them and their SHA-256 checksums, records whether Scholar still offered more
rows, and marks coverage as
`complete_visible_profile` or `capped_partial`. The former means every row
visible on that profile was captured when the Show more button stopped; it does
not claim every work in the researcher's academic history is listed on Scholar.
An access refusal retains the previous good cache and is recorded separately.
