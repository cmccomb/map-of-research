# Pegden and Bohman Scholar observations — 2026-10-24

Direct Chrome captures of the CMU-linked Google Scholar profiles show all 89
visible rows for Wesley Pegden and all 90 for Tom Bohman. In each saved page,
the **Show more** button was disabled and no page error was visible. The paired
HTML/JSON captures and canonical caches are staged on the snapshot branch
`codex/math-pegden-bohman-snapshot`; each Scholar citation ID and bibliography
field was checked against the saved page. These observations are a complete
visible profile capture, not a claim that every row is a publication.

The reviewed exclusion manifest records 13 NSF awards (three Pegden, ten
Bohman). Each Scholar row includes an NSF award number. The official
[NSF award API](https://api.nsf.gov/services/v1/awards.json?id=2054503)
returned the same award ID and title for every one of the 13 rows; the manifest
links directly to each award. An award is funding metadata, not a published
research work. The manifest also excludes Pegden's Springer front-matter
title, which duplicates the separately retained 2016 article by Levine,
Pegden, and Smart. [MIT's publication record](https://dspace.mit.edu/entities/publication/015ced8f-c94c-4173-8343-04d8323423fd)
confirms the article, author list, journal, volume, and pages.

Other rows, including manuscripts, theses, meeting abstracts, and lecture
notes, remain as Scholar observations pending any specific contrary evidence.
The raw snapshot retains every captured row, including reviewed exclusions;
the publisher applies the manifest to map eligibility by exact Scholar row ID.
