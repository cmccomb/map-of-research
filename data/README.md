# Collector cache

The `automation/map-snapshot` branch stores one atomic JSON cache file per
unique Scholar profile under `data/authors/`. The default branch owns the
trusted pipeline code and source definitions.

Cache records retain the last good normalized source observation. Registry role
changes never delete caches; excluded and former relationships remain available
to the lossless dataset pipeline.

For new direct browser captures, store matching HTML and extracted row JSON in
`data/captures/`. The Fonseca completion and the seven Biological Sciences
captures each include both files. Their cache and `status/browser-capture-*.json`
receipt name the files and SHA-256 checksums, record whether Scholar still
offered more rows, and mark coverage as
`complete_visible_profile` or `capped_partial`. The former means every row
visible on that profile was captured when the Show more button stopped; it does
not claim every work in the researcher's academic history is listed on Scholar.
An access refusal retains the previous good cache and is recorded separately.

The first Biological Sciences batch adds 565 visible Scholar rows from seven
complete profiles: Aryn Gittis (85), Catherine Armbruster (52), Drew Bridges
(27), Russell Schwartz (307), Irene Kaplow (49), Jonathan Henninger (27), and
Kate Hong (18). The raw records remain intact when exact reviewed exclusions
remove nonpublications or false attributions from mapped works.

The second Biological Sciences batch adds 686 visible rows from six complete
profiles: Alison Barth (115), Eric Yttri (51), Kenneth Hovis (6), Huaiying
Zhang (61), Yongxin Zhao (86), and Robert F. Murphy (367). Each cache keeps the
Scholar row and citation identifier, and each paired capture preserves the
source evidence after replacing captured form values with `[REDACTED]`.
Publication eligibility is reviewed separately from raw-source preservation.

The third Biological Sciences batch adds 228 visible rows from two complete
profiles: Jonathan Minden (127) and Luisa Hiller (101). Scholar displays the
latter account as “Hiller NL”; its CMU-linked identity is recorded in the
registry review. The captured rows remain attributed to that Scholar account
and are available for separate publication review.

The fourth Biological Sciences batch adds 79 visible rows from two complete
profiles: Zheng Kuang (41) and Tina Lee (38). Scholar displays the latter
account as “Tina H Lee”; its verified CMU identity is recorded in the registry
review. Their raw HTML and JSON, normalized caches, and checksum receipts are
staged together. Exact publication eligibility is reviewed separately.

The next Biological Sciences source capture preserves all 142 visible rows on
Gordon Rule's direct Scholar profile, with paired redacted HTML and JSON, a
cache, and a checksum receipt. His official CMU Biological Sciences roster
identity is known, but this isolated branch does not assign the Scholar ID in
the registry. The cache is not eligible for a full-corpus snapshot until that
identity review is complete. Exact nonpublication and attribution decisions
remain a separate source review.

The next local Biological Sciences batch adds 81 complete visible Scholar rows:
C. Joel McManus (62, displayed as “Joel McManus”) and Elizabeth Ransey (19).
Their paired redacted HTML and JSON, normalized caches, and checksum receipts
are staged together. The Scholar IDs and official CMU identity pages are
verified on current producer main, while this stacked source branch predates
those registry assignments. The frozen 15-profile release does not include
these caches. Exact nonpublication and attribution decisions remain a separate
source review.
