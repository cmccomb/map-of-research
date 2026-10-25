# Pipeline receipts

The automation branch stores small machine-readable receipts here:

- `authors.json`: per-profile attempt/success state used for bounded rotation;
- `last-collection.json`: the latest Scholar-facing collection outcome;
- `browser-capture-YYYY-MM-DD.json`: a dated receipt when directly linked
  profiles are captured in a bounded local browser session, including source
  checksums and whether a profile stopped at the publication cap;
- `last-upload.json`: the latest Hugging Face publication outcome, including
  the immutable current Hub revision, region comparison revision, exact
  producer and snapshot commits, and region release-diff receipt;
- `last-successful-upload.json`: the last successful upload receipt, retained
  across failed attempts to pin the next trusted upload.

Receipts contain public profile IDs and operational metadata, never credentials
or cookies. Failed collection never deletes the last good profile cache.
