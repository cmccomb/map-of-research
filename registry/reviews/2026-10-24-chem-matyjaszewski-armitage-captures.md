# Chemistry Scholar source captures: Matyjaszewski and Armitage

These files preserve every row visible after the two verified Google Scholar
profiles' **Show more** controls became disabled on 2026-10-24. The paired
HTML and JSON were checked row by row before conversion to full author caches.
This source batch does not approve each Scholar row as a publication or update
the map snapshot. Exact nonpublication and attribution decisions require a
separate review.

| CMU Chemistry faculty member | Scholar identity evidence | Visible rows | Profile expansion |
| --- | --- | ---: | --- |
| [Krzysztof Matyjaszewski](https://www.cmu.edu/chemistry/people/faculty/matyjaszewski.html) | CMU faculty page links [`TSkli5EAAAAJ`](https://scholar.google.com/citations?user=TSkli5EAAAAJ&hl=en). | 2,707 | 29 page loads, including 28 Show more clicks; final button disabled. |
| [Bruce Armitage](https://www.cmu.edu/chemistry/people/faculty/armitage.html) | [CMU Experts profile](https://experts.cmu.edu/expert/bruce.armitage/bruce-armitage) links [`BWNUifwAAAAJ`](https://scholar.google.com/citations?user=BWNUifwAAAAJ&hl=en). Scholar displays “Bruce A. Armitage”; the registry displays “Bruce Armitage.” | 207 | Four page loads, including three Show more clicks; final button disabled. |

Both captures used the isolated task browser, encountered no challenge or page
error, and ended with the task tabs closed and the shared Scholar lane released.
The paired source HTML contains `[REDACTED]` in both hidden `xsrf` values for
each profile; redaction left all visible publication rows unchanged. The
converter confirmed contiguous row numbers, exact HTML/JSON field agreement,
unique Scholar citation IDs, the final disabled button, and no malformed
nonempty years. It preserved 136 blank Matyjaszewski years and one blank
Armitage year without inference; starred citation displays remain in the
cache's `citation_count_display` field.

| Profile | JSON SHA-256 | Redacted HTML SHA-256 | Author cache SHA-256 |
| --- | --- | --- | --- |
| `TSkli5EAAAAJ` | `0a6baf6edf0003cb4b28fa42f4240fd9279de3e0adb5da474b972f690c1408a9` | `4fbcbf2e8786d9243caaff17644ac1a16c9070dd158956d7f6c589394df3685e` | `970c1698164515449f88673fa52ee7072bd1525b13600c05de4ba3ecd8e42146` |
| `BWNUifwAAAAJ` | `f366e9eb4f249ddbee7afb64d885aa6bd7bc379ffa0654b1b61837f0a1ed1d1d` | `40a8851ef69d039b614def8a4f484938507d56c90dd6aebd336e65acac0a6be5` | `4f7b6930b478965fb7b0a7e2b42921730b6bbc891fe3305b5465afbba371c8e8` |

Per-profile receipts at `status/browser-capture-chem-{matyjaszewski,armitage}-2026-10-24.json`
record the source paths, checksums, capture times, row counts, browser route,
and completeness state. No canonical snapshot or manifest is changed here.
