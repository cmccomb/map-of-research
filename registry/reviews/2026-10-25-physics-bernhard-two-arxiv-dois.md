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
