# Chien Ho Scholar context review

This draft policy is limited to 12 exact Scholar observation IDs from Chien
Ho's fully captured 582-row profile. The raw cache and raw snapshot remain
unchanged. The [exact context manifest](../../data/curated/scholar-contexts.json)
records each source URL, reason, and date.

| Scholar row | Context | Research map | Collaboration map |
| --- | --- | --- | --- |
| 116 | Edited book | Include as a reviewed book | No coauthor edge between editors |
| 292, 328, 359, 394, 406, 424, 426, 446, 451 | Patent document | Include as a reviewed patent | No coauthor edge between inventors |
| 425 | Withdrawn article | Retain raw observation; exclude mapped work | No edge |
| 525 | After-dinner panel transcript | Retain raw observation; exclude mapped work | No edge |

The patent work key uses the exact publication number in the source manifest.
The [parent patent US6486123B1](https://patents.google.com/patent/US6486123B1/en)
and [published application US20020151468A1](https://patents.google.com/patent/US20020151468A1/en)
both have the Scholar title *Low oxygen affinity mutant hemoglobins* and year
2002. Title-year matching would incorrectly collapse rows 359 and 394. They
remain separate published documents even though they belong to a related
application family. Rows 424 and 426 also have separate patent publication
numbers. A family relationship is not itself a coauthor relationship.

The [CMU bibliography](https://www.cmu.edu/bio/people/faculty/images/ho_publications.pdf)
lists row 116 under books and names all eight editors. The Scholar byline lists
only C. Ho and a misspelled W. A. Eaton, so the reviewed manifest carries the
complete display list while the raw and canonical Scholar authors stay intact.
The [PubMed withdrawal
record](https://pubmed.ncbi.nlm.nih.gov/15194702/) for row 425 says
not all coauthors approved submission. The Crossref publisher-deposited record
for row 525 names Chien Ho as a panel participant rather than a research-paper
coauthor; its exact source URL is in the manifest.
The two unmapped observations carry `map_eligible: false` in their exact
manifest entries; the panel decision does not exclude every future panel
transcript by type.

## Release checks

1. Verify the Chien exact positive-attribution gate and all remaining held
   observations independently. This context policy does not attest to the
   other profile rows.
2. Confirm the immutable snapshot still contains exactly 582 Chien raw rows,
   including all 12 context rows, and that rows 425 and 525 are absent from
   mapped works.
3. Confirm patent rows 359 and 394 have distinct `work_id` values and their
   exact publication numbers; no patent, editor, or panel edge may appear in
   the collaboration graph.
4. Show the reviewed type and the proper inventor or editor label on the
   research map before releasing the 10 mapped non-article records (nine
   patents and one edited book). The present `map-of-eng` details view labels
   all contributor strings “Authors,” which would misdescribe inventors and
   editors.
5. Check the producer manifest hash in the upload receipt, then verify both
   rendered maps against the same immutable dataset revision.

The policy adds no external inventor/editor/speaker nodes. Any future graph of
those relationships needs a separate edge type and source-backed identity
records.
