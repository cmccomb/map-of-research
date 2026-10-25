# Stefan Bernhard: frozen combined snapshot review

This candidate pairs producer `main` at
`c60096988b0fe55a76a123c2de03592f6f8dd18f` with source
`automation/map-snapshot` at `ab825f48a22451fe2964365f36ce83ced6c413c9`.
The source branch already contains the complete Bernhard capture merged in
#190 and the previous trusted upload receipt. The producer contains the
Bernhard primary Chemistry home, 43 exact reviewed exclusions, one provisional
hold, and complete-profile release decision merged through #191 and #246.

The isolated frozen-pair build produced 51,823 raw observations from 310
complete or legacy cached Scholar profiles and 26 preserved verified curated
rows. Its Parquet SHA-256 is
`5b940614b2238c124bc194e944da277ee2221df27a11eedca6693af020ee86fb`.
The previous 51,595 raw rows match exactly, including all raw fields; the 228
added rows exactly equal the Bernhard cache. The snapshot registry contains the
producer's evidenced primary `chem` assignment. The upload workflow is
byte-identical to the previous trusted release.

The local no-upload gates passed: full Scholar coverage (24 complete, 286
legacy, 12 reconciled prior IDs), Math coverage (36 complete profiles, 4,080
rows, 29 unresolved Scholar IDs), and 487 evidenced primary academic homes.
The Bernhard cache contains 228 distinct exact publication IDs and declares a
complete visible profile with no more rows. Profile release and provisional
hold checks bind its SHA-256 and exact source IDs.

The offline relationship preview used the producer's normal correction,
release, attribution, alias, exclusion, context, and provisional-hold rules,
with synthetic embeddings and no upload. It yielded 47,693 works and 49,662
authorships before embedding, layout, topic, and low-information-region
processing. All 49,478 authorships from the previous release remain identical.
Of Bernhard's 228 rows, 43 have exact reviewed exclusions, one has an exact
provisional hold, and 184 contribute eligible authorships. The publisher's
held-contribution checks passed. The machine-readable preview is at
`work/second-mcs-release-prep/candidate-c60096988b0f-ab825f48a224/no-upload-preview.json`
in the task workspace.

This draft is for independent review. Merging a snapshot PR targeting
`automation/map-snapshot` triggers the trusted upload; keep the frozen producer
head unchanged through upload and receipt verification. No upload or map deploy
was performed for this candidate.
