# Validated snapshots

The automation branch stores the source-observation Parquet snapshot and
checksum manifest here. Scholar profile rows have `source_type=scholar` and a
real Scholar fetch timestamp. Separately verified DOI observations have
`source_type=curated_verified`, a verification date and URLs, and no Scholar
fetch timestamp. The manifest binds the registry and curated-work file
checksums before publication. Raw snapshots contain no Hugging Face credentials
or model code.
