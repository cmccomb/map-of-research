# Physics: exact arXiv DOI review for two Bernhard region rows

The frozen source snapshot retains these two Scholar observations with blank
`source_doi` fields. Both are genuine works by the named CMU Physics faculty.
This producer review adds exact, evidence-backed DOI identities for the **observed
preprints**. It does not edit the raw Scholar HTML, JSON, cache, Parquet, or
`source_record_json`, and does not publish a new dataset or map.

| Scholar observation | Observed work | Exact DOI and attribution evidence |
| --- | --- | --- |
| Colin Morningstar `8AqWr_4AAAAJ:XvxMoLDsR5gC`, saved row 94 | *Toward a resolution of the NN controversy*, 2021, arXiv:2112.04569 | [arXiv](https://arxiv.org/abs/2112.04569) names Morningstar and assigns preprint DOI `10.48550/arXiv.2112.04569`. The [2022 Proceedings of Science paper](https://findresearcher.sdu.dk/ws/portalfiles/portal/204594796/LATTICE2021_098.pdf) lists his CMU Physics affiliation. Its DOI `10.22323/1.396.0098` belongs to that related final publication and is **not** used for this 2021 observation. |
| Diana Parno `JbSQsXAAAAAJ:2osOgNQ5qMEC`, saved row 56 | *LGBT+ inclusivity in physics and astronomy: A best practices guide*, 2018, arXiv:1804.08406 | [arXiv](https://arxiv.org/abs/1804.08406) names Parno, identifies this 85-page second edition, and assigns DOI `10.48550/arXiv.1804.08406`. [CMU Physics](https://www.cmu.edu/physics/ci/ci-resources.html) independently calls Parno a coauthor of this edition. |

The DOI-alias manifest guards each exact Scholar ID, captured title, year, and
venue; requires the matching arXiv abstract URL; and records the review date
and independent attribution evidence. The preprint DOI may change the derived
`work_id`, so release comparison must follow the exact observation IDs and
check every historical visibility and topic decision against the pinned live
Hub revision `bb88ddf0d427122cbb8d72cfed5c84c4e6d660bd`.

## Pending map placement

The exact DOI aliases are retained in the derived work and observation rows,
but the two new DOI works remain off the map with work-level reason
`topic_placement_pending`. The held work IDs are
`work-b33b19863dc6eaeb1919045ad850ae25ba419ab998ecb8738ec7c79174afffa8`
and `work-e40eef6b5a2616eae3f6a2868d2dbea1b8b842f514d3e2c76a5c9b0fb4cb4309`.
Each hold requires its exact Scholar citation, observation ID, title, year,
preprint DOI, and reviewed alias. It changes the work's map eligibility after
layout and before topic assignment; the source observation and authorship row
remain intact. Publication of either point needs reviewed topic and spatial
placement.

The pinned bb88 layout also contains Noonan's `Fullerenes` work as a
`low_information_region` outlier. Its later reviewed false-attribution
exclusion is the sole prior-work mismatch in the frozen Physics corpus. The
producer permits reuse only for that exact unchanged work ID and source
identity, and omits its old coordinate. A later run can reuse the two Physics
works' coordinates when their exact pending holds are reapplied.

On current producer main `68ab26ed`, the source branch `9bb7474` still pins a
26-row curated snapshot while producer main has 28 curated rows; its upload
preflight must not bypass this mismatch. A disposable local rebuild from the
`9bb7474` cache and current curated records produced 51,879 rows, but the
ordinary current-main region gate stopped its no-upload replay on 22 Sullivan
observation losses before any Hub write. The separate Sullivan release gate is
under review in draft PR #316. A combined Sullivan/Physics release therefore
needs its own exact-head source and map review; the frozen Physics-only
no-upload proof does not authorize that release.
