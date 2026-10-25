# Pipeline receipts

The automation branch stores small machine-readable receipts here:

- `authors.json`: per-profile attempt/success state used for bounded rotation;
- `last-collection.json`: the latest Scholar-facing collection outcome;
- `browser-capture-*.json`: a dated receipt when directly linked profiles are
  captured in a bounded local browser session, including source checksums,
  row count, and whether Scholar still offered more rows;
- `last-upload.json`: the latest Hugging Face publication outcome.

Receipts contain public profile IDs and operational metadata, never credentials
or cookies. Failed collection never deletes the last good profile cache.

A `browser-capture-chem-donahue-2026-10-24.json` receipt records the complete
visible Donahue capture and its original `staged_for_source_review` state.
The paired disappeared-ID and suspect-row CSVs document the review before
the 623-row candidate was copied to the canonical cache. The receipt is a
point-in-time capture record; a later snapshot build and upload need their
own receipts.
