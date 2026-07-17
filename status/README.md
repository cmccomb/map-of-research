# Pipeline receipts

The automation branch stores small machine-readable receipts here:

- `authors.json`: per-profile attempt/success state used for bounded rotation;
- `last-collection.json`: the latest Scholar-facing collection outcome;
- `last-upload.json`: the latest Hugging Face publication outcome.

Receipts contain public profile IDs and operational metadata, never credentials
or cookies. Failed collection never deletes the last good profile cache.
