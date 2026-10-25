# Scholar cache coverage baseline — 2026-10-25

The baseline in `registry/legacy-scholar-cache-baseline.json` is pinned to the
immutable source snapshot commit
`4dbafc2a627930a0ef80aa2c645089546c0d874a` and snapshot SHA-256
`69b495c07e4d1ac7b630af80656d8cf7c0882f99bf37395d6971451b159c0768`.
It records exact cache-file hashes and 49,687 Scholar citation IDs for 288
profiles in the raw snapshot. Of those profiles, 262 had an included
membership. A cache absent from the baseline, changed from its frozen hash,
or newly included after exclusion must carry a complete visible-profile
browser receipt with saved JSON and HTML evidence. None of the legacy hashes
is a completeness claim.

The gate checks the source caches before `map-build-snapshot` writes a new
Parquet file. The trusted upload checks the same receipts and reconciles every
Scholar `source_record_json` row in the candidate Parquet file to its cache
before obtaining a Hugging Face token or writing a dataset config. It rejects
removal of a frozen citation ID or a frozen cache. This permits the existing
legacy corpus to remain visible while each person receives a reviewed full
capture.

The Kietrys and Gil caches are not in the frozen baseline. On merged source
tip `c3995740`, their 36 and 195 rows, along with an updated complete Jarvik
capture, passed source JSON/HTML hash, disabled Show More, citation ID, year,
and citation count checks. The source tip still contains the prior 49,708-row
Parquet snapshot and a registry file older than producer main `29d0a415`; its
checked-in snapshot fails registry validation. A read-only rebuild from that
exact source tip and current producer registry yielded 49,939 raw rows and
290 cached profiles: 287 unchanged legacy and 3 complete. Both the Math and
whole-profile gates passed on the rebuilt preview. This draft did not commit
that preview or change a dataset or public map.

After a newly completed profile is approved and published, add its exact cache
hash and citation IDs to this baseline from the immutable published source
commit. The baseline is reviewed source data, not a collector output or a
moving branch pointer. It must never be regenerated from a partial or failed
collection. The completeness claim remains limited to the visible Scholar
profile, not publications absent from Scholar.

The staged Neil M. Donahue promotion illustrates a deliberate stop. Its
complete browser cache has 623 rows, sharing 539 citation IDs with the 551-row
legacy cache; 84 IDs are new and 12 frozen IDs are absent. The reviewed
academic works can be retained through rekeyed Scholar records or separately
verified curated records, but the gate cannot infer those equivalences from
titles. It blocks that promotion until a source-backed, exact 12-ID crosswalk
or a reviewed exception proves where each prior observation went. Do not
remove those IDs from the baseline merely to pass the gate. Kietrys and Gil
have no frozen cache IDs and are unaffected.
