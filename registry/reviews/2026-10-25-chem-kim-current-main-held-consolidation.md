# Hyung J. Kim held review on current main

This local candidate starts at main `5275d5aecd82eb0ac6cdf8d9136568f08ae2d517`
and selectively carries the Kim review through local `f47658ae`. The 918-row
raw Scholar capture remains in merged source PR #118, with saved JSON SHA-256
`ea04610e120039df7aaf83c402d193021f328b319cd41cfabf541f3942069867`.
The capture reports no remaining visible profile rows. It is a complete
visible Scholar profile, not a guarantee of every publication in Kim's career.

The [screening ledger](2026-10-24-chem-kim-918-screen.csv),
[exact exclusion proposal](proposals/scholar-exclusions.kim-proposal.json),
[attribution proposal](proposals/scholar-attribution-reviews.kim-proposal.json),
and [release proposal](proposals/scholar-profile-releases.kim-proposal.json)
partition the 918 exact citation IDs into **123 confirmed CMU works, 533 exact
exclusions, and 262 held**. The eight earlier exclusions and 525 later exact
decisions are all in the Kim-only proposal. Four earlier decisions are
documented in the [exact follow-up](2026-10-25-chem-kim-four-more-held-exact-attributions.md);
the subsequent [AMoRE-I ORCID check](2026-10-25-chem-kim-amore-i-orcid-exact.md)
resolves one additional citation. Three further HongJoo Kim author links are
documented in the [next-twelve review](2026-10-25-chem-kim-next-twelve-exact-links.md).
Four more exact author links are documented in the
[following review](2026-10-25-chem-kim-next-four-exact-author-links.md).
Six more named-person or complete-byline decisions are documented in the
[mid-list review](2026-10-25-chem-kim-six-mid-list-exact-attributions.md).
[Nine further KNU person-linked records](2026-10-25-chem-kim-nine-knu-person-links.md)
resolve only their exact Scholar IDs; eight nearby rows remain held.
[Nine later exact namesake records](2026-10-25-chem-kim-late-nine-exact-namesakes.md)
resolve only their exact Scholar IDs; the related JCO abstract and other
initials-only neighboring rows remain held.
[Four later KNU person links and self-asserted ORCID works](2026-10-25-chem-kim-late-four-person-links.md)
resolve only their exact Scholar IDs; the other 18 checked rows remain held.
[Six further late-list KNU, ORCID, and original-report links](2026-10-25-chem-kim-final-six-knu-2008-source-links.md)
resolve six exact Scholar IDs; twelve other checked candidates remain held.
Row 809 remains held. The draft PR #347 body reports older counts; these
checked fragments supersede its counts.

Main's 1,286 canonical exclusions, including 31 Collins decisions, retain
their original records, order, and bytes. The canonical exclusion digest is
`adb6532806737f794d02ea1fed30ad9e03a419abb27b819a7d6a01cbf3122f66`;
the Sullivan/Math historical gates pin that exact file. The Kim exclusions
proposal is inert and kept outside `data/curated`. Main's collector,
publisher, profile release and atomic write code is untouched. Kim has no
entry in the active profile release or attribution manifests and no canonical
`data/authors/bC4aq-4AAAAJ.json` cache. His whole profile remains off both
maps.

The [rebuilder](rebuild_chem_kim_918_review.py) checks the saved capture
against the Kim-only exclusion proposal and ledger. The
[offline preparer](prepare_chem_kim_reviewed_subset.py) checks the same
partition, fails if Kim has become active in the canonical exclusions, and
regenerates proposal fragments and an isolated candidate cache. Its cache
SHA-256 remains
`f4b7b37efcd3310a05ff82057488ecb286450e3db0dc390b4e26b125fc378b2e`.
No upload or public release follows from either script.
