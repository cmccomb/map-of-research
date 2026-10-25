# Reconcile an interrupted atomic dataset release

The upload workflow commits `status/last-upload.json` to `automation/map-snapshot`
before calling the Hugging Face Hub. Its `atomic_release_intent` records the exact
parent revision, producer and source commits, prewrite gate, commit nonce, and
the path, byte size, and SHA-256 of all seven staged files. The checkpoint is a
failure receipt until immutable Hub verification completes. It leaves
`status/last-successful-upload.json` untouched.

If the Hub call or a later verification step fails, read `atomic_release_intent`
and `atomic_release_reconciliation` from the current run's receipt. Do not retry
the upload while its write outcome is unknown.

1. Find the candidate Hub commit by the exact `commit_nonce` in its commit
   message. `artifact_revision` and `observed_head_revision`, when present, are
   additional clues. Require exactly one matching commit. `HfApi.list_repo_commits`
   exposes commit IDs and messages but does not expose parent IDs.
2. Read the candidate's Git commit object and confirm its **only parent** is
   `parent_revision` from the intent. A read-only, no-checkout clone of the
   dataset Git repository can provide the object; `git cat-file -p <candidate>`
   prints its `parent` line. A matching nonce without this ancestry check is
   insufficient.
3. Download each of the seven `committed_files` at the candidate's immutable
   revision, then compare every byte size and SHA-256 with `size_by_path` and
   `sha256_by_path` in the intent. Check the current Hub head separately. A
   matching file set proves what landed at that revision; a later head needs
   its own review.
4. Record a reviewed `atomic_release_resolution` on the failure receipt,
   binding its `commit_nonce` and `parent_revision` to the exact intent. Include
   `reviewer`, `reviewed_at_utc`, and an HTTPS `evidence_url` for the read-only
   evidence. For a confirmed no-write, use `outcome: reconciled_no_write`,
   `matching_commit_ids: []`, and `observed_head_revision` equal to the parent.
   For a confirmed release, use `outcome: verified_success`,
   `artifact_revision`, and `verified_sha256_by_path` and
   `verified_size_by_path` equal to all seven intent values. Advance
   `last-successful-upload.json` to that verified revision through the same
   reviewed change. If no unique direct child passes every comparison, keep
   the unresolved failure receipt.

The upload uses `parent_commit` and a non-forced GitHub checkpoint push. A
stale parent or moved snapshot branch stops the next write. This protects
against an unsafe retry but does not substitute for the read-only reconciliation
above. `pin_prior_release` rejects an unresolved intent even when the Hub still
appears to be at the old parent. A reviewed no-write outcome permits a fresh
run; a verified release outcome requires the successful baseline to match it.
