# Validated snapshots

The automation branch stores the source-observation Parquet snapshot and
checksum manifest here. Scholar profile rows have `source_type=scholar` and a
real Scholar fetch timestamp. The two manually verified Tetali–Erdős DOI rows
have `source_type=curated_verified`, a verification date and URLs, and no
Scholar fetch timestamp. The manifest binds the registry and curated-work file
checksums before publication. Raw snapshots contain no Hugging Face credentials
or model code.

When author-owned bibliography captures are selected, repeat
`--author-site-capture` once per person. The manifest's ordered
`author_site_captures` list binds each person ID, manifest filename, SHA-256,
and full row count. Validation reloads every pinned bibliography and compares
every author-site snapshot row to its captured source. Duplicate people,
manifest filenames, and source-native IDs are rejected. The older singular
`author_site_capture_file` and `author_site_capture_sha256` fields remain for
zero- and one-capture compatibility; legacy schema-v7 manifests without the
new list remain valid in those cases. The publisher still blocks
`source_type=author_site` until a separate reviewed release gate.
