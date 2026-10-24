# Pipeline receipts

The automation branch stores small machine-readable receipts here:

- `authors.json`: per-profile attempt/success state used for bounded rotation;
- `last-collection.json`: the latest Scholar-facing collection outcome;
- `browser-capture-YYYY-MM-DD.json`: a dated receipt when directly linked
  profiles are captured in a bounded local browser session, including source
  checksums and whether a profile stopped at the publication cap;
- `last-upload.json`: the latest Hugging Face publication outcome.

Receipts contain public profile IDs and operational metadata, never credentials
or cookies. Failed collection never deletes the last good profile cache.
